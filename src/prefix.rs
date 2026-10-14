use anyhow::{anyhow, Result};

/// Longest prefix the parser accepts: the whole 32-byte public key. This is
/// not a policy limit: every character multiplies the expected search by 16
/// (on an RTX 5090: 9 characters about 6 s, 10 about 1.5 minutes, 12 about
/// 6.5 hours, on average), so callers that serve others cap it themselves
/// (`ServerConfig::max_prefix_chars`, the time limit, or their own rules).
pub const MAX_PREFIX_CHARS: usize = 64;

/// A MeshCore public-key hex prefix (1 to [`MAX_PREFIX_CHARS`] characters,
/// case-insensitive). MeshCore reserves keys whose first byte is 0x00 or
/// 0xFF, so those prefixes are rejected and such keys never match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prefix {
    text: String,
    bytes: Vec<u8>,
    nibbles: usize,
}

impl Prefix {
    pub fn parse(input: &str) -> Result<Self> {
        let text = input.trim().to_ascii_uppercase();
        if text.is_empty() {
            return Err(anyhow!("prefix must not be empty"));
        }
        if text.len() > MAX_PREFIX_CHARS {
            return Err(anyhow!(
                "prefix must be 1 to {MAX_PREFIX_CHARS} hex characters (got {})",
                text.len()
            ));
        }
        if !text.as_bytes().iter().all(|b| b.is_ascii_hexdigit()) {
            return Err(anyhow!("prefix must contain only hex characters 0-9/A-F"));
        }
        if text.len() >= 2 && (text.starts_with("00") || text.starts_with("FF")) {
            return Err(anyhow!(
                "prefixes starting with 00 or FF are reserved by MeshCore"
            ));
        }

        let nibbles = text.len();
        let mut bytes = vec![0u8; nibbles.div_ceil(2)];
        for (i, ch) in text.bytes().enumerate() {
            let nibble = match ch {
                b'0'..=b'9' => ch - b'0',
                b'A'..=b'F' => ch - b'A' + 10,
                _ => unreachable!(),
            };
            if i & 1 == 0 {
                bytes[i >> 1] = nibble << 4;
            } else {
                bytes[i >> 1] |= nibble;
            }
        }

        Ok(Self {
            text,
            bytes,
            nibbles,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn nibbles(&self) -> usize {
        self.nibbles
    }

    /// All prefix bytes, most significant hex pair first (i.e. in public-key
    /// byte order). An odd trailing nibble occupies the high half of the last
    /// byte.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The prefix bytes packed little-endian. The GPU compares this against
    /// the low 8 bytes of the canonical public key (its fast path handles up
    /// to 16 hex characters; the host re-checks every candidate against the
    /// whole prefix, so longer prefixes are filtered there).
    pub fn packed_u64(&self) -> u64 {
        let mut packed = 0u64;
        for (i, byte) in self.bytes.iter().copied().enumerate().take(8) {
            packed |= (byte as u64) << (8 * i);
        }
        packed
    }

    /// All prefix bytes packed into eight little-endian u32 words (byte i in
    /// word i/4 at bit 8*(i%4)); unused trailing bytes are zero. This is the
    /// layout of `SearchParams::prefix_full` on the GPU.
    pub fn packed_words(&self) -> [u32; 8] {
        let mut words = [0u32; 8];
        for (i, &byte) in self.bytes.iter().enumerate() {
            words[i / 4] |= (byte as u32) << (8 * (i % 4));
        }
        words
    }

    pub fn matches(&self, public_key: &[u8; 32]) -> bool {
        if public_key[0] == 0x00 || public_key[0] == 0xFF {
            return false;
        }

        let full_bytes = self.nibbles / 2;
        for i in 0..full_bytes {
            if public_key[i] != self.bytes[i] {
                return false;
            }
        }

        if self.nibbles & 1 == 1 {
            let i = full_bytes;
            if (public_key[i] & 0xF0) != (self.bytes[i] & 0xF0) {
                return false;
            }
        }

        true
    }

    /// Match against the y coordinate alone (bit 255, the sign of x, is
    /// not known yet). Equivalent to `matches` for prefixes up to 62
    /// nibbles, i.e. for every prefix this crate accepts; kept general so
    /// the walk callers can stay oblivious to the cap.
    #[inline]
    pub fn matches_y(&self, y: &[u8; 32]) -> bool {
        if y[0] == 0x00 || y[0] == 0xFF {
            return false;
        }
        if y[0] != self.bytes[0] && self.nibbles >= 2 {
            return false;
        }

        let full_bytes = self.nibbles / 2;
        for i in 1..full_bytes.min(31) {
            if y[i] != self.bytes[i] {
                return false;
            }
        }
        if full_bytes == 32 {
            return (y[31] & 0x7F) == (self.bytes[31] & 0x7F);
        }

        if self.nibbles & 1 == 1 {
            let i = full_bytes;
            let mask = if i == 31 { 0x70 } else { 0xF0 };
            if (y[i] & mask) != (self.bytes[i] & mask) {
                return false;
            }
        }

        true
    }

    /// Keys expected per hit.
    pub fn expected_attempts(&self) -> f64 {
        1.0 / self.match_probability()
    }

    pub fn match_probability(&self) -> f64 {
        if self.nibbles == 1 && (self.text == "0" || self.text == "F") {
            15.0 / 256.0
        } else {
            16f64.powi(-(self.nibbles as i32))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Prefix;

    #[test]
    fn parses_odd_nibble_prefix() {
        let prefix = Prefix::parse("Fa1").unwrap();
        assert_eq!(prefix.text(), "FA1");
        assert_eq!(prefix.nibbles(), 3);
        assert_eq!(prefix.packed_u64() & 0xFFFF, 0x10FA);
    }

    #[test]
    fn rejects_reserved_full_byte_prefixes() {
        assert!(Prefix::parse("00").is_err());
        assert!(Prefix::parse("ffab").is_err());
        assert!(Prefix::parse("0").is_ok());
    }

    #[test]
    fn accepts_prefixes_up_to_the_whole_key() {
        let prefix = Prefix::parse("a1b2c3d4").unwrap();
        assert_eq!(prefix.nibbles(), 8);
        assert_eq!(prefix.bytes(), &[0xA1, 0xB2, 0xC3, 0xD4]);
        assert_eq!(prefix.packed_u64(), 0xD4C3_B2A1);
        let words = prefix.packed_words();
        assert_eq!(words[0], 0xD4C3_B2A1);
        assert_eq!(words[1..], [0u32; 7]);
        assert_eq!(prefix.expected_attempts(), 4294967296.0);
        let prefix = Prefix::parse("a1b2c3d4e").unwrap();
        assert_eq!(prefix.nibbles(), 9);
        assert_eq!(prefix.bytes(), &[0xA1, 0xB2, 0xC3, 0xD4, 0xE0]);
        assert_eq!(prefix.packed_u64(), 0xE0_D4C3_B2A1);
        assert_eq!(prefix.packed_words()[..2], [0xD4C3_B2A1, 0xE0]);
        assert_eq!(prefix.expected_attempts(), 68719476736.0);
        let whole = "a1".repeat(32);
        let prefix = Prefix::parse(&whole).unwrap();
        assert_eq!(prefix.nibbles(), 64);
        assert_eq!(prefix.bytes(), &[0xA1; 32]);
        assert_eq!(prefix.packed_words(), [0xA1A1_A1A1; 8]);
        let err = Prefix::parse(&format!("{whole}0")).unwrap_err().to_string();
        assert!(err.contains("1 to 64 hex characters"), "{err}");
        assert!(Prefix::parse("").is_err());
        assert!(Prefix::parse("g1").is_err());
        assert!(Prefix::parse(" 1a ").is_ok());
    }

    #[test]
    fn matches_y_is_matches_with_unknown_sign_bit() {
        let mut key = [0u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(13) ^ 0xA5;
        }
        key[0] = 0x3C;
        let hex_key = hex::encode_upper(key);
        let mut y = key;
        y[31] &= 0x7F;
        let mut y_signed = key;
        y_signed[31] |= 0x80;

        for len in 1..=8usize {
            let prefix = Prefix::parse(&hex_key[..len]).unwrap();
            assert!(prefix.matches_y(&y), "len {len}");
            assert!(prefix.matches_y(&y_signed), "len {len} signed");
            assert_eq!(prefix.matches_y(&y), prefix.matches(&y));
            assert_eq!(prefix.matches_y(&y), prefix.matches(&y_signed));
            // Corrupt the last prefix nibble: must not match.
            let mut wrong = key;
            if len % 2 == 0 {
                wrong[len / 2 - 1] ^= 0x01;
            } else {
                wrong[len / 2] ^= 0x10;
            }
            assert!(!prefix.matches_y(&wrong), "len {len} corrupt");
            assert!(!prefix.matches(&wrong), "len {len} corrupt");
            if len > 1 {
                let mut wrong = key;
                wrong[0] ^= 0x01;
                assert!(!prefix.matches_y(&wrong), "len {len} corrupt first byte");
            }
        }

        // Reserved first bytes are rejected exactly like `matches`.
        let prefix = Prefix::parse("0").unwrap();
        let mut zero = [0u8; 32];
        assert!(!prefix.matches_y(&zero) && !prefix.matches(&zero));
        zero[0] = 0x01;
        assert!(prefix.matches_y(&zero));
        let prefix = Prefix::parse("F").unwrap();
        let mut ff = [0xFFu8; 32];
        assert!(!prefix.matches_y(&ff) && !prefix.matches(&ff));
        ff[0] = 0xFE;
        assert!(prefix.matches_y(&ff));
        // The single-nibble reserved prefixes have 15/256 of keys available.
        assert_eq!(Prefix::parse("0").unwrap().match_probability(), 15.0 / 256.0);
        assert_eq!(Prefix::parse("7").unwrap().match_probability(), 1.0 / 16.0);
    }

    #[test]
    fn matches_public_key_bytes() {
        let prefix = Prefix::parse("ABC").unwrap();
        let mut key = [0u8; 32];
        key[0] = 0xAB;
        key[1] = 0xC5;
        assert!(prefix.matches(&key));
        key[1] = 0xB5;
        assert!(!prefix.matches(&key));
    }
}
