enum HNSWDistanceComputers {
    protocol Computer {
        static func distanceBetween<Stored: HNSWScalar>(
            _ lhsID: HNSWInternalID,
            _ rhsID: HNSWInternalID,
            storage: UnsafeBufferPointer<Stored>,
            dimensions: Int
        ) -> Float

        static func distanceFromQuery<Stored: HNSWScalar>(
            _ query: UnsafeBufferPointer<Stored>,
            to internalID: HNSWInternalID,
            storage: UnsafeBufferPointer<Stored>,
            dimensions: Int
        ) -> Float
    }

    enum L2: Computer {
        @inline(__always)
        @_specialize(where Stored == Float)
        @_specialize(where Stored == Float16)
        static func distanceBetween<Stored: HNSWScalar>(
            _ lhsID: HNSWInternalID,
            _ rhsID: HNSWInternalID,
            storage: UnsafeBufferPointer<Stored>,
            dimensions: Int
        ) -> Float {
            let lhsOffset = Int(lhsID) * dimensions
            let rhsOffset = Int(rhsID) * dimensions
            return VectorOperations.squaredL2ComparisonDistance(
                from: storage.baseAddress! + lhsOffset,
                to: storage.baseAddress! + rhsOffset,
                count: dimensions
            )
        }

        @inline(__always)
        @_specialize(where Stored == Float)
        @_specialize(where Stored == Float16)
        static func distanceFromQuery<Stored: HNSWScalar>(
            _ query: UnsafeBufferPointer<Stored>,
            to internalID: HNSWInternalID,
            storage: UnsafeBufferPointer<Stored>,
            dimensions: Int
        ) -> Float {
            let offset = Int(internalID) * dimensions
            return VectorOperations.squaredL2ComparisonDistance(
                from: query.baseAddress!,
                to: storage.baseAddress! + offset,
                count: dimensions
            )
        }
    }

    enum InnerProduct: Computer {
        @inline(__always)
        @_specialize(where Stored == Float)
        @_specialize(where Stored == Float16)
        static func distanceBetween<Stored: HNSWScalar>(
            _ lhsID: HNSWInternalID,
            _ rhsID: HNSWInternalID,
            storage: UnsafeBufferPointer<Stored>,
            dimensions: Int
        ) -> Float {
            let lhsOffset = Int(lhsID) * dimensions
            let rhsOffset = Int(rhsID) * dimensions
            return VectorOperations.innerProductComparisonDistance(
                from: storage.baseAddress! + lhsOffset,
                to: storage.baseAddress! + rhsOffset,
                count: dimensions
            )
        }

        @inline(__always)
        @_specialize(where Stored == Float)
        @_specialize(where Stored == Float16)
        static func distanceFromQuery<Stored: HNSWScalar>(
            _ query: UnsafeBufferPointer<Stored>,
            to internalID: HNSWInternalID,
            storage: UnsafeBufferPointer<Stored>,
            dimensions: Int
        ) -> Float {
            let offset = Int(internalID) * dimensions
            return VectorOperations.innerProductComparisonDistance(
                from: query.baseAddress!,
                to: storage.baseAddress! + offset,
                count: dimensions
            )
        }
    }
}
