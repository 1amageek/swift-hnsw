/// Efficient packing and unpacking of b-bit quantization indices into bytes.
///
/// Packing formats:
/// - b=1: 8 indices per byte
/// - b=2: 4 indices per byte
/// - b=3: 8 indices per 3 bytes (24 bits)
/// - b=4: 2 indices per byte (nibble)
enum BitPacking {

    /// Compute the number of bytes needed to store `count` indices at `bitWidth` bits each.
    static func packedSize(count: Int, bitWidth: Int) -> Int {
        (count * bitWidth + 7) / 8
    }

    /// Pack an array of b-bit indices into a compact byte array.
    ///
    /// - Parameters:
    ///   - indices: Array of indices, each in range [0, 2^bitWidth).
    ///   - bitWidth: Bits per index (0, 1, 2, 3, or 4).
    /// - Returns: Packed byte array.
    static func pack(_ indices: [UInt8], bitWidth: Int) -> [UInt8] {
        guard bitWidth > 0 else {
            return []
        }
        let count = indices.count
        let outputSize = packedSize(count: count, bitWidth: bitWidth)
        var output = [UInt8](repeating: 0, count: outputSize)

        switch bitWidth {
        case 4:
            packNibbles(indices, into: &output)
        case 2:
            packTwoBit(indices, into: &output)
        case 1:
            packOneBit(indices, into: &output)
        case 3:
            packThreeBit(indices, into: &output)
        default:
            packGeneric(indices, bitWidth: bitWidth, into: &output)
        }

        return output
    }

    /// Unpack a compact byte array back into an array of b-bit indices.
    ///
    /// - Parameters:
    ///   - packed: Packed byte array.
    ///   - bitWidth: Bits per index (1, 2, 3, or 4).
    ///   - count: Number of indices to unpack.
    /// - Returns: Array of unpacked indices.
    static func unpack(_ packed: [UInt8], bitWidth: Int, count: Int) -> [UInt8] {
        guard bitWidth > 0 else {
            return [UInt8](repeating: 0, count: count)
        }
        var output = [UInt8](repeating: 0, count: count)

        switch bitWidth {
        case 4:
            unpackNibbles(packed, into: &output, count: count)
        case 2:
            unpackTwoBit(packed, into: &output, count: count)
        case 1:
            unpackOneBit(packed, into: &output, count: count)
        case 3:
            unpackThreeBit(packed, into: &output, count: count)
        default:
            unpackGeneric(packed, bitWidth: bitWidth, into: &output, count: count)
        }

        return output
    }

    /// Reads one packed index without allocating an intermediate unpacked buffer.
    ///
    /// The bit order matches the format used by `pack`: 1-, 2-, and 4-bit values are
    /// stored most-significant first within each byte; 3-bit values use the historical
    /// little-endian bit stream so that eight values occupy three bytes.
    @inline(__always)
    static func value(
        at index: Int,
        from packed: UnsafeBufferPointer<UInt8>,
        bitWidth: Int
    ) -> UInt8 {
        switch bitWidth {
        case 0:
            return 0
        case 4:
            let byte = packed[index >> 1]
            return index & 1 == 0 ? byte >> 4 : byte & 0x0F
        case 2:
            let byte = packed[index >> 2]
            let shift = 6 - ((index & 3) * 2)
            return (byte >> shift) & 0x03
        case 1:
            let byte = packed[index >> 3]
            return (byte >> (7 - (index & 7))) & 0x01
        case 3:
            let bitOffset = index * 3
            let byteIndex = bitOffset >> 3
            let bitInByte = bitOffset & 7
            var value = packed[byteIndex] >> bitInByte
            if bitInByte > 5 {
                value |= packed[byteIndex + 1] << (8 - bitInByte)
            }
            return value & 0x07
        default:
            let mask = UInt8((1 << bitWidth) - 1)
            let bitOffset = index * bitWidth
            let byteIndex = bitOffset >> 3
            let bitInByte = bitOffset & 7
            var value = packed[byteIndex] >> bitInByte
            if bitInByte + bitWidth > 8 {
                value |= packed[byteIndex + 1] << (8 - bitInByte)
            }
            return value & mask
        }
    }

    /// Stores one index in an already zero-initialized packed buffer.
    ///
    /// This is the allocation-free insertion primitive used by the production quantizer.
    /// The caller must write each logical index at most once.
    @inline(__always)
    static func store(
        _ value: UInt8,
        at index: Int,
        in packed: UnsafeMutableBufferPointer<UInt8>,
        bitWidth: Int
    ) {
        switch bitWidth {
        case 0:
            return
        case 4:
            let byteIndex = index >> 1
            if index & 1 == 0 {
                packed[byteIndex] |= (value & 0x0F) << 4
            } else {
                packed[byteIndex] |= value & 0x0F
            }
        case 2:
            let byteIndex = index >> 2
            packed[byteIndex] |= (value & 0x03) << (6 - (index & 3) * 2)
        case 1:
            let byteIndex = index >> 3
            packed[byteIndex] |= (value & 0x01) << (7 - (index & 7))
        case 3:
            let bitOffset = index * 3
            let byteIndex = bitOffset >> 3
            let bitInByte = bitOffset & 7
            let storedValue = value & 0x07
            packed[byteIndex] |= storedValue << bitInByte
            if bitInByte > 5 {
                packed[byteIndex + 1] |= storedValue >> (8 - bitInByte)
            }
        default:
            let mask = UInt8((1 << bitWidth) - 1)
            let bitOffset = index * bitWidth
            let byteIndex = bitOffset >> 3
            let bitInByte = bitOffset & 7
            let storedValue = value & mask
            packed[byteIndex] |= storedValue << bitInByte
            if bitInByte + bitWidth > 8 {
                packed[byteIndex + 1] |= storedValue >> (8 - bitInByte)
            }
        }
    }

    // MARK: - 4-bit (Nibble) Packing

    private static func packNibbles(_ indices: [UInt8], into output: inout [UInt8]) {
        let pairs = indices.count / 2
        for i in 0..<pairs {
            output[i] = (indices[2 * i] << 4) | (indices[2 * i + 1] & 0x0F)
        }
        if indices.count % 2 != 0 {
            output[pairs] = indices[indices.count - 1] << 4
        }
    }

    private static func unpackNibbles(_ packed: [UInt8], into output: inout [UInt8], count: Int) {
        let pairs = count / 2
        for i in 0..<pairs {
            output[2 * i] = packed[i] >> 4
            output[2 * i + 1] = packed[i] & 0x0F
        }
        if count % 2 != 0 {
            output[count - 1] = packed[pairs] >> 4
        }
    }

    // MARK: - 2-bit Packing

    private static func packTwoBit(_ indices: [UInt8], into output: inout [UInt8]) {
        let quads = indices.count / 4
        for i in 0..<quads {
            let base = 4 * i
            output[i] = (indices[base] << 6)
                | ((indices[base + 1] & 0x03) << 4)
                | ((indices[base + 2] & 0x03) << 2)
                | (indices[base + 3] & 0x03)
        }
        let remaining = indices.count % 4
        if remaining > 0 {
            var byte: UInt8 = 0
            for j in 0..<remaining {
                byte |= (indices[quads * 4 + j] & 0x03) << (6 - j * 2)
            }
            output[quads] = byte
        }
    }

    private static func unpackTwoBit(_ packed: [UInt8], into output: inout [UInt8], count: Int) {
        let quads = count / 4
        for i in 0..<quads {
            let byte = packed[i]
            output[4 * i] = (byte >> 6) & 0x03
            output[4 * i + 1] = (byte >> 4) & 0x03
            output[4 * i + 2] = (byte >> 2) & 0x03
            output[4 * i + 3] = byte & 0x03
        }
        let remaining = count % 4
        if remaining > 0 {
            let byte = packed[quads]
            for j in 0..<remaining {
                output[quads * 4 + j] = (byte >> (6 - j * 2)) & 0x03
            }
        }
    }

    // MARK: - 1-bit Packing

    private static func packOneBit(_ indices: [UInt8], into output: inout [UInt8]) {
        let octets = indices.count / 8
        for i in 0..<octets {
            let base = 8 * i
            var byte: UInt8 = 0
            for j in 0..<8 {
                byte |= (indices[base + j] & 0x01) << (7 - j)
            }
            output[i] = byte
        }
        let remaining = indices.count % 8
        if remaining > 0 {
            var byte: UInt8 = 0
            for j in 0..<remaining {
                byte |= (indices[octets * 8 + j] & 0x01) << (7 - j)
            }
            output[octets] = byte
        }
    }

    private static func unpackOneBit(_ packed: [UInt8], into output: inout [UInt8], count: Int) {
        let octets = count / 8
        for i in 0..<octets {
            let byte = packed[i]
            for j in 0..<8 {
                output[8 * i + j] = (byte >> (7 - j)) & 0x01
            }
        }
        let remaining = count % 8
        if remaining > 0 {
            let byte = packed[octets]
            for j in 0..<remaining {
                output[octets * 8 + j] = (byte >> (7 - j)) & 0x01
            }
        }
    }

    // MARK: - 3-bit Packing

    private static func packThreeBit(_ indices: [UInt8], into output: inout [UInt8]) {
        var bitOffset = 0
        for idx in indices {
            let byteIndex = bitOffset / 8
            let bitInByte = bitOffset % 8
            let val = idx & 0x07
            output[byteIndex] |= val << bitInByte
            if bitInByte > 5 {
                // Spans two bytes
                output[byteIndex + 1] |= val >> (8 - bitInByte)
            }
            bitOffset += 3
        }
    }

    private static func unpackThreeBit(_ packed: [UInt8], into output: inout [UInt8], count: Int) {
        var bitOffset = 0
        for i in 0..<count {
            let byteIndex = bitOffset / 8
            let bitInByte = bitOffset % 8
            var val = (packed[byteIndex] >> bitInByte) & 0x07
            if bitInByte > 5 && byteIndex + 1 < packed.count {
                val |= (packed[byteIndex + 1] << (8 - bitInByte)) & 0x07
            }
            output[i] = val
            bitOffset += 3
        }
    }

    // MARK: - Generic Packing

    private static func packGeneric(_ indices: [UInt8], bitWidth: Int, into output: inout [UInt8]) {
        let mask = UInt8((1 << bitWidth) - 1)
        var bitOffset = 0
        for idx in indices {
            let byteIndex = bitOffset / 8
            let bitInByte = bitOffset % 8
            let val = idx & mask
            output[byteIndex] |= val << bitInByte
            if bitInByte + bitWidth > 8 && byteIndex + 1 < output.count {
                output[byteIndex + 1] |= val >> (8 - bitInByte)
            }
            bitOffset += bitWidth
        }
    }

    private static func unpackGeneric(_ packed: [UInt8], bitWidth: Int, into output: inout [UInt8], count: Int) {
        let mask = UInt8((1 << bitWidth) - 1)
        var bitOffset = 0
        for i in 0..<count {
            let byteIndex = bitOffset / 8
            let bitInByte = bitOffset % 8
            var val = (packed[byteIndex] >> bitInByte)
            if bitInByte + bitWidth > 8 && byteIndex + 1 < packed.count {
                val |= packed[byteIndex + 1] << (8 - bitInByte)
            }
            output[i] = val & mask
            bitOffset += bitWidth
        }
    }
}
