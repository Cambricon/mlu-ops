// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEMATRIX_H
#define EIGEN_SPARSEMATRIX_H

#include "./InternalHeaderCheck.h"

namespace Eigen { 

/** \ingroup SparseCore_Module
  *
  * \class SparseMatrix
  *
  * \brief A versatible sparse matrix representation
  *
  * This class implements a more versatile variants of the common \em compressed row/column storage format.
  * Each colmun's (resp. row) non zeros are stored as a pair of value with associated row (resp. colmiun) index.
  * All the non zeros are stored in a single large buffer. Unlike the \em compressed format, there might be extra
  * space in between the nonzeros of two successive colmuns (resp. rows) such that insertion of new non-zero
  * can be done with limited memory reallocation and copies.
  *
  * A call to the function makeCompressed() turns the matrix into the standard \em compressed format
  * compatible with many library.
  *
  * More details on this storage sceheme are given in the \ref TutorialSparse "manual pages".
  *
  * \tparam Scalar_ the scalar type, i.e. the type of the coefficients
  * \tparam Options_ Union of bit flags controlling the storage scheme. Currently the only possibility
  *                 is ColMajor or RowMajor. The default is 0 which means column-major.
  * \tparam StorageIndex_ the type of the indices. It has to be a \b signed type (e.g., short, int, std::ptrdiff_t). Default is \c int.
  *
  * \warning In %Eigen 3.2, the undocumented type \c SparseMatrix::Index was improperly defined as the storage index type (e.g., int),
  *          whereas it is now (starting from %Eigen 3.3) deprecated and always defined as Eigen::Index.
  *          Codes making use of \c SparseMatrix::Index, might thus likely have to be changed to use \c SparseMatrix::StorageIndex instead.
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizing_Plugins by defining the preprocessor symbol \c EIGEN_SPARSEMATRIX_PLUGIN.
  */

namespace internal {
template<typename Scalar_, int Options_, typename StorageIndex_>
struct traits<SparseMatrix<Scalar_, Options_, StorageIndex_> >
{
  typedef Scalar_ Scalar;
  typedef StorageIndex_ StorageIndex;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = Options_ | NestByRefBit | LvalueBit | CompressedAccessBit,
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

template<typename Scalar_, int Options_, typename StorageIndex_, int DiagIndex>
struct traits<Diagonal<SparseMatrix<Scalar_, Options_, StorageIndex_>, DiagIndex> >
{
  typedef SparseMatrix<Scalar_, Options_, StorageIndex_> MatrixType;
  typedef typename ref_selector<MatrixType>::type MatrixTypeNested;
  typedef std::remove_reference_t<MatrixTypeNested> MatrixTypeNested_;

  typedef Scalar_ Scalar;
  typedef Dense StorageKind;
  typedef StorageIndex_ StorageIndex;
  typedef MatrixXpr XprKind;

  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = 1,
    Flags = LvalueBit
  };
};

template<typename Scalar_, int Options_, typename StorageIndex_, int DiagIndex>
struct traits<Diagonal<const SparseMatrix<Scalar_, Options_, StorageIndex_>, DiagIndex> >
 : public traits<Diagonal<SparseMatrix<Scalar_, Options_, StorageIndex_>, DiagIndex> >
{
  enum {
    Flags = 0
  };
};

template <typename StorageIndex>
struct sparse_reserve_op {
  EIGEN_DEVICE_FUNC sparse_reserve_op(Index begin, Index end, Index size) {
    Index range = numext::mini(end - begin, size);
    m_begin = begin;
    m_end = begin + range;
    m_val = StorageIndex(size / range);
    m_remainder = StorageIndex(size % range);
  }
  template <typename IndexType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE StorageIndex operator()(IndexType i) const {
    if ((i >= m_begin) && (i < m_end))
      return m_val + ((i - m_begin) < m_remainder ? 1 : 0);
    else
      return 0;
  }
  StorageIndex m_val, m_remainder;
  Index m_begin, m_end;
};

template <typename Scalar>
struct functor_traits<sparse_reserve_op<Scalar>> {
  enum { Cost = 1, PacketAccess = false, IsRepeatable = true };
};

} // end namespace internal

template<typename Scalar_, int Options_, typename StorageIndex_>
class SparseMatrix
  : public SparseCompressedBase<SparseMatrix<Scalar_, Options_, StorageIndex_> >
{
    typedef SparseCompressedBase<SparseMatrix> Base;
    using Base::convert_index;
    friend class SparseVector<Scalar_,0,StorageIndex_>;
    template<typename, typename, typename, typename, typename>
    friend struct internal::Assignment;
  public:
    using Base::isCompressed;
    using Base::nonZeros;
    EIGEN_SPARSE_PUBLIC_INTERFACE(SparseMatrix)
    using Base::operator+=;
    using Base::operator-=;

    typedef Eigen::Map<SparseMatrix<Scalar,Flags,StorageIndex>> Map;
    typedef Diagonal<SparseMatrix> DiagonalReturnType;
    typedef Diagonal<const SparseMatrix> ConstDiagonalReturnType;
    typedef typename Base::InnerIterator InnerIterator;
    typedef typename Base::ReverseInnerIterator ReverseInnerIterator;
    

    using Base::IsRowMajor;
    typedef internal::CompressedStorage<Scalar,StorageIndex> Storage;
    enum {
      Options = Options_
    };

    typedef typename Base::IndexVector IndexVector;
    typedef typename Base::ScalarVector ScalarVector;
  protected:
    typedef SparseMatrix<Scalar,(Flags&~RowMajorBit)|(IsRowMajor?RowMajorBit:0),StorageIndex> TransposedSparseMatrix;

    Index m_outerSize;
    Index m_innerSize;
    StorageIndex* m_outerIndex;
    StorageIndex* m_innerNonZeros;     // optional, if null then the data is compressed
    Storage m_data;

  public:
    
    /** \returns the number of rows of the matrix */
    inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
    /** \returns the number of columns of the matrix */
    inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }

    /** \returns the number of rows (resp. columns) of the matrix if the storage order column major (resp. row major) */
    inline Index innerSize() const { return m_innerSize; }
    /** \returns the number of columns (resp. rows) of the matrix if the storage order column major (resp. row major) */
    inline Index outerSize() const { return m_outerSize; }
    
    /** \returns a const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline const Scalar* valuePtr() const { return m_data.valuePtr(); }
    /** \returns a non-const pointer to the array of values.
      * This function is aimed at interoperability with other libraries.
      * \sa innerIndexPtr(), outerIndexPtr() */
    inline Scalar* valuePtr() { return m_data.valuePtr(); }

    /** \returns a const pointer to the array of inner indices.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), outerIndexPtr() */
    inline const StorageIndex* innerIndexPtr() const { return m_data.indexPtr(); }
    /** \returns a non-const pointer to the array of inner indices.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), outerIndexPtr() */
    inline StorageIndex* innerIndexPtr() { return m_data.indexPtr(); }

    /** \returns a const pointer to the array of the starting positions of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), innerIndexPtr() */
    inline const StorageIndex* outerIndexPtr() const { return m_outerIndex; }
    /** \returns a non-const pointer to the array of the starting positions of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \sa valuePtr(), innerIndexPtr() */
    inline StorageIndex* outerIndexPtr() { return m_outerIndex; }

    /** \returns a const pointer to the array of the number of non zeros of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \warning it returns the null pointer 0 in compressed mode */
    inline const StorageIndex* innerNonZeroPtr() const { return m_innerNonZeros; }
    /** \returns a non-const pointer to the array of the number of non zeros of the inner vectors.
      * This function is aimed at interoperability with other libraries.
      * \warning it returns the null pointer 0 in compressed mode */
    inline StorageIndex* innerNonZeroPtr() { return m_innerNonZeros; }

    /** \internal */
    inline Storage& data() { return m_data; }
    /** \internal */
    inline const Storage& data() const { return m_data; }

    /** \returns the value of the matrix at position \a i, \a j
      * This function returns Scalar(0) if the element is an explicit \em zero */
    inline Scalar coeff(Index row, Index col) const
    {
      eigen_assert(row>=0 && row<rows() && col>=0 && col<cols());
      
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer+1];
      return m_data.atInRange(m_outerIndex[outer], end, inner);
    }

    /** \returns a non-const reference to the value of the matrix at position \a i, \a j
      *
      * If the element does not exist then it is inserted via the insert(Index,Index) function
      * which itself turns the matrix into a non compressed form if that was not the case.
      *
      * This is a O(log(nnz_j)) operation (binary search) plus the cost of insert(Index,Index)
      * function if the element does not already exist.
      */
    inline Scalar& coeffRef(Index row, Index col)
    {
      eigen_assert(row>=0 && row<rows() && col>=0 && col<cols());
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;
      Index start = m_outerIndex[outer];
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer + 1];
      eigen_assert(end >= start && "you probably called coeffRef on a non finalized matrix");
      if (end <= start) return insertAtByOuterInner(outer, inner, start);
      Index dst = m_data.searchLowerIndex(start, end, inner);
      if ((dst < end) && (m_data.index(dst) == inner))
        return m_data.value(dst);
      else
        return insertAtByOuterInner(outer, inner, dst);
    }

    /** \returns a reference to a novel non zero coefficient with coordinates \a row x \a col.
      * The non zero coefficient must \b not already exist.
      *
      * If the matrix \c *this is in compressed mode, then \c *this is turned into uncompressed
      * mode while reserving room for 2 x this->innerSize() non zeros if reserve(Index) has not been called earlier.
      * In this case, the insertion procedure is optimized for a \e sequential insertion mode where elements are assumed to be
      * inserted by increasing outer-indices.
      * 
      * If that's not the case, then it is strongly recommended to either use a triplet-list to assemble the matrix, or to first
      * call reserve(const SizesType &) to reserve the appropriate number of non-zero elements per inner vector.
      *
      * Assuming memory has been appropriately reserved, this function performs a sorted insertion in O(1)
      * if the elements of each inner vector are inserted in increasing inner index order, and in O(nnz_j) for a random insertion.
      *
      */
    inline Scalar& insert(Index row, Index col);

  public:

    /** Removes all non zeros but keep allocated memory
      *
      * This function does not free the currently allocated memory. To release as much as memory as possible,
      * call \code mat.data().squeeze(); \endcode after resizing it.
      * 
      * \sa resize(Index,Index), data()
      */
    inline void setZero()
    {
      m_data.clear();
      std::fill_n(m_outerIndex, m_outerSize + 1, StorageIndex(0));
      if(m_innerNonZeros) {
        std::fill_n(m_innerNonZeros, m_outerSize, StorageIndex(0));
      }
    }

    /** Preallocates \a reserveSize non zeros.
      *
      * Precondition: the matrix must be in compressed mode. */
    inline void reserve(Index reserveSize)
    {
      eigen_assert(isCompressed() && "This function does not make sense in non compressed mode.");
      m_data.reserve(reserveSize);
    }
    
    #ifdef EIGEN_PARSED_BY_DOXYGEN
    /** Preallocates \a reserveSize[\c j] non zeros for each column (resp. row) \c j.
      *
      * This function turns the matrix in non-compressed mode.
      * 
      * The type \c SizesType must expose the following interface:
        \code
        typedef value_type;
        const value_type& operator[](i) const;
        \endcode
      * for \c i in the [0,this->outerSize()[ range.
      * Typical choices include std::vector<int>, Eigen::VectorXi, Eigen::VectorXi::Constant, etc.
      */
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes);
    #else
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes, const typename SizesType::value_type& enableif =
        typename SizesType::value_type())
    {
      EIGEN_UNUSED_VARIABLE(enableif);
      reserveInnerVectors(reserveSizes);
    }
    #endif // EIGEN_PARSED_BY_DOXYGEN
  protected:
    template<class SizesType>
    inline void reserveInnerVectors(const SizesType& reserveSizes)
    {
      if(isCompressed())
      {
        Index totalReserveSize = 0;
        for (Index j = 0; j < m_outerSize; ++j) totalReserveSize += reserveSizes[j];

        // if reserveSizes is empty, don't do anything!
        if (totalReserveSize == 0) return;

        // turn the matrix into non-compressed mode
        m_innerNonZeros = internal::conditional_aligned_new_auto<StorageIndex, true>(m_outerSize);
        
        // temporarily use m_innerSizes to hold the new starting points.
        StorageIndex* newOuterIndex = m_innerNonZeros;
        
        StorageIndex count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          count += reserveSizes[j] + (m_outerIndex[j+1]-m_outerIndex[j]);
        }

        m_data.reserve(totalReserveSize);
        StorageIndex previousOuterIndex = m_outerIndex[m_outerSize];
        for(Index j=m_outerSize-1; j>=0; --j)
        {
          StorageIndex innerNNZ = previousOuterIndex - m_outerIndex[j];
          StorageIndex begin = m_outerIndex[j];
          StorageIndex end = begin + innerNNZ;
          StorageIndex target = newOuterIndex[j];
          internal::smart_memmove(innerIndexPtr() + begin, innerIndexPtr() + end, innerIndexPtr() + target);
          internal::smart_memmove(valuePtr() + begin, valuePtr() + end, valuePtr() + target);
          previousOuterIndex = m_outerIndex[j];
          m_outerIndex[j] = newOuterIndex[j];
          m_innerNonZeros[j] = innerNNZ;
        }
        if(m_outerSize>0)
          m_outerIndex[m_outerSize] = m_outerIndex[m_outerSize-1] + m_innerNonZeros[m_outerSize-1] + reserveSizes[m_outerSize-1];
        
        m_data.resize(m_outerIndex[m_outerSize]);
      }
      else
      {
        StorageIndex* newOuterIndex = internal::conditional_aligned_new_auto<StorageIndex, true>(m_outerSize + 1);
        
        StorageIndex count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          StorageIndex alreadyReserved = (m_outerIndex[j+1]-m_outerIndex[j]) - m_innerNonZeros[j];
          StorageIndex toReserve = std::max<StorageIndex>(reserveSizes[j], alreadyReserved);
          count += toReserve + m_innerNonZeros[j];
        }
        newOuterIndex[m_outerSize] = count;
        
        m_data.resize(count);
        for(Index j=m_outerSize-1; j>=0; --j)
        {
          StorageIndex offset = newOuterIndex[j] - m_outerIndex[j];
          if(offset>0)
          {
            StorageIndex innerNNZ = m_innerNonZeros[j];
            StorageIndex begin = m_outerIndex[j];
            StorageIndex end = begin + innerNNZ;
            StorageIndex target = newOuterIndex[j];
            internal::smart_memmove(innerIndexPtr() + begin, innerIndexPtr() + end, innerIndexPtr() + target);
            internal::smart_memmove(valuePtr() + begin, valuePtr() + end, valuePtr() + target);
          }
        }
        
        std::swap(m_outerIndex, newOuterIndex);
        internal::conditional_aligned_delete_auto<StorageIndex, true>(newOuterIndex, m_outerSize + 1);
      }
      
    }
  public:

    //--- low level purely coherent filling ---

    /** \internal
      * \returns a reference to the non zero coefficient at position \a row, \a col assuming that:
      * - the nonzero does not already exist
      * - the new coefficient is the last one according to the storage order
      *
      * Before filling a given inner vector you must call the statVec(Index) function.
      *
      * After an insertion session, you should call the finalize() function.
      *
      * \sa insert, insertBackByOuterInner, startVec */
    inline Scalar& insertBack(Index row, Index col)
    {
      return insertBackByOuterInner(IsRowMajor?row:col, IsRowMajor?col:row);
    }

    /** \internal
      * \sa insertBack, startVec */
    inline Scalar& insertBackByOuterInner(Index outer, Index inner)
    {
      eigen_assert(Index(m_outerIndex[outer+1]) == m_data.size() && "Invalid ordered insertion (invalid outer index)");
      eigen_assert( (m_outerIndex[outer+1]-m_outerIndex[outer]==0 || m_data.index(m_data.size()-1)<inner) && "Invalid ordered insertion (invalid inner index)");
      StorageIndex p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(Scalar(0), inner);
      return m_data.value(p);
    }

    /** \internal
      * \warning use it only if you know what you are doing */
    inline Scalar& insertBackByOuterInnerUnordered(Index outer, Index inner)
    {
      StorageIndex p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(Scalar(0), inner);
      return m_data.value(p);
    }

    /** \internal
      * \sa insertBack, insertBackByOuterInner */
    inline void startVec(Index outer)
    {
      eigen_assert(m_outerIndex[outer]==Index(m_data.size()) && "You must call startVec for each inner vector sequentially");
      eigen_assert(m_outerIndex[outer+1]==0 && "You must call startVec for each inner vector sequentially");
      m_outerIndex[outer+1] = m_outerIndex[outer];
    }

    /** \internal
      * Must be called after inserting a set of non zero entries using the low level compressed API.
      */
    inline void finalize()
    {
      if(isCompressed())
      {
        StorageIndex size = internal::convert_index<StorageIndex>(m_data.size());
        Index i = m_outerSize;
        // find the last filled column
        while (i>=0 && m_outerIndex[i]==0)
          --i;
        ++i;
        while (i<=m_outerSize)
        {
          m_outerIndex[i] = size;
          ++i;
        }
      }
    }

    //---

    template<typename InputIterators>
    void setFromTriplets(const InputIterators& begin, const InputIterators& end);

    template<typename InputIterators,typename DupFunctor>
    void setFromTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

    template<typename Derived, typename DupFunctor>
    void collapseDuplicates(DenseBase<Derived>& wi, DupFunctor dup_func = DupFunctor());

    template<typename InputIterators>
    void setFromSortedTriplets(const InputIterators& begin, const InputIterators& end);

    template<typename InputIterators, typename DupFunctor>
    void setFromSortedTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func);

    //---
    
    /** \internal
      * same as insert(Index,Index) except that the indices are given relative to the storage order */
    Scalar& insertByOuterInner(Index j, Index i)
    {
      return insert(IsRowMajor ? j : i, IsRowMajor ? i : j);
    }

    /** Turns the matrix into the \em compressed format.
      */
    void makeCompressed()
    {
      if (isCompressed()) return;
      
      eigen_internal_assert(m_outerIndex!=0 && m_outerSize>0);
      
      StorageIndex start = m_outerIndex[1];
      m_outerIndex[1] = m_innerNonZeros[0];
      for(Index j=1; j<m_outerSize; ++j)
      {
        StorageIndex end = start + m_innerNonZeros[j];
        StorageIndex target = m_outerIndex[j];
        if (start != target)
        {
          internal::smart_memmove(innerIndexPtr() + start, innerIndexPtr() + end, innerIndexPtr() + target);
          internal::smart_memmove(valuePtr() + start, valuePtr() + end, valuePtr() + target);
        }
        start = m_outerIndex[j + 1];
        m_outerIndex[j + 1] = m_outerIndex[j] + m_innerNonZeros[j];
      }
      internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
      m_innerNonZeros = 0;
      m_data.resize(m_outerIndex[m_outerSize]);
      m_data.squeeze();
    }

    /** Turns the matrix into the uncompressed mode */
    void uncompress()
    {
      if(m_innerNonZeros != 0) return; 
      m_innerNonZeros = internal::conditional_aligned_new_auto<StorageIndex, true>(m_outerSize);
      typename IndexVector::AlignedMapType innerNonZeroMap(m_innerNonZeros, m_outerSize);
      typename IndexVector::ConstAlignedMapType outerIndexMap(m_outerIndex, m_outerSize);
      typename IndexVector::ConstMapType nextOuterIndexMap(m_outerIndex + 1, m_outerSize);
      innerNonZeroMap = nextOuterIndexMap - outerIndexMap;
    }

    /** Suppresses all nonzeros which are \b much \b smaller \b than \a reference under the tolerance \a epsilon */
    void prune(const Scalar& reference, const RealScalar& epsilon = NumTraits<RealScalar>::dummy_precision())
    {
      prune(default_prunning_func(reference,epsilon));
    }
    
    /** Turns the matrix into compressed format, and suppresses all nonzeros which do not satisfy the predicate \a keep.
      * The functor type \a KeepFunc must implement the following function:
      * \code
      * bool operator() (const Index& row, const Index& col, const Scalar& value) const;
      * \endcode
      * \sa prune(Scalar,RealScalar)
      */
    template<typename KeepFunc>
    void prune(const KeepFunc& keep = KeepFunc())
    {
      StorageIndex k = 0;
      for(Index j=0; j<m_outerSize; ++j)
      {
        StorageIndex previousStart = m_outerIndex[j];
        if (isCompressed())
          m_outerIndex[j] = k;
        else
          k = m_outerIndex[j];
        StorageIndex end = isCompressed() ? m_outerIndex[j+1] : previousStart + m_innerNonZeros[j];
        for(StorageIndex i=previousStart; i<end; ++i)
        {
          StorageIndex row = IsRowMajor ? StorageIndex(j) : m_data.index(i);
          StorageIndex col = IsRowMajor ? m_data.index(i) : StorageIndex(j);
          bool keepEntry = keep(row, col, m_data.value(i));
          if (keepEntry) {
            m_data.value(k) = m_data.value(i);
            m_data.index(k) = m_data.index(i);
            ++k;
          } else if (!isCompressed())
            m_innerNonZeros[j]--;
        }
      }
      if (isCompressed()) {
        m_outerIndex[m_outerSize] = k;
        m_data.resize(k, 0);
      }
    }

    /** Resizes the matrix to a \a rows x \a cols matrix leaving old values untouched.
      *
      * If the sizes of the matrix are decreased, then the matrix is turned to \b uncompressed-mode
      * and the storage of the out of bounds coefficients is kept and reserved.
      * Call makeCompressed() to pack the entries and squeeze extra memory.
      *
      * \sa reserve(), setZero(), makeCompressed()
      */
    void conservativeResize(Index rows, Index cols) {

      // If one dimension is null, then there is nothing to be preserved
      if (rows == 0 || cols == 0) return resize(rows, cols);

      Index newOuterSize = IsRowMajor ? rows : cols;
      Index newInnerSize = IsRowMajor ? cols : rows;

      Index innerChange = newInnerSize - innerSize();
      Index outerChange = newOuterSize - outerSize();

      if (outerChange != 0) {
        m_outerIndex = internal::conditional_aligned_realloc_new_auto<StorageIndex, true>(
            outerIndexPtr(), newOuterSize + 1, outerSize() + 1);

        if (!isCompressed())
          m_innerNonZeros = internal::conditional_aligned_realloc_new_auto<StorageIndex, true>(
              innerNonZeroPtr(), newOuterSize, outerSize());

        if (outerChange > 0) {
          StorageIndex lastIdx = outerSize() == 0 ? StorageIndex(0) : outerIndexPtr()[outerSize()];
          std::fill_n(outerIndexPtr() + outerSize(), outerChange + 1, lastIdx);

          if (!isCompressed()) std::fill_n(innerNonZeroPtr() + outerSize(), outerChange, StorageIndex(0));
        }
      }
      m_outerSize = newOuterSize;

      if (innerChange < 0) {
        for (Index j = 0; j < outerSize(); j++) {
          Index start = outerIndexPtr()[j];
          Index end = isCompressed() ? outerIndexPtr()[j + 1] : start + innerNonZeroPtr()[j];
          Index lb = data().searchLowerIndex(start, end, newInnerSize);
          if (lb != end) {
            uncompress();
            innerNonZeroPtr()[j] = StorageIndex(lb - start);
          }
        }
      }
      m_innerSize = newInnerSize;

      Index newSize = outerIndexPtr()[outerSize()];
      eigen_assert(newSize <= m_data.size());
      m_data.resize(newSize);
    }
    
    /** Resizes the matrix to a \a rows x \a cols matrix and initializes it to zero.
      * 
      * This function does not free the currently allocated memory. To release as much as memory as possible,
      * call \code mat.data().squeeze(); \endcode after resizing it.
      * 
      * \sa reserve(), setZero()
      */
    void resize(Index rows, Index cols)
    {
      const Index outerSize = IsRowMajor ? rows : cols;
      m_innerSize = IsRowMajor ? cols : rows;
      m_data.clear();
      if (m_outerSize != outerSize || m_outerSize==0)
      {
        m_outerIndex = internal::conditional_aligned_realloc_new_auto<StorageIndex, true>(m_outerIndex, outerSize + 1,
            m_outerSize + 1);
        m_outerSize = outerSize;
      }
      if(m_innerNonZeros)
      {
        internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
        m_innerNonZeros = 0;
      }
      std::fill_n(m_outerIndex, m_outerSize + 1, StorageIndex(0));
    }

    /** \internal
      * Resize the nonzero vector to \a size */
    void resizeNonZeros(Index size)
    {
      m_data.resize(size);
    }

    /** \returns a const expression of the diagonal coefficients. */
    const ConstDiagonalReturnType diagonal() const { return ConstDiagonalReturnType(*this); }
    
    /** \returns a read-write expression of the diagonal coefficients.
      * \warning If the diagonal entries are written, then all diagonal
      * entries \b must already exist, otherwise an assertion will be raised.
      */
    DiagonalReturnType diagonal() { return DiagonalReturnType(*this); }

    /** Default constructor yielding an empty \c 0 \c x \c 0 matrix */
    inline SparseMatrix()
      : m_outerSize(-1), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      resize(0, 0);
    }

    /** Constructs a \a rows \c x \a cols empty matrix */
    inline SparseMatrix(Index rows, Index cols)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      resize(rows, cols);
    }

    /** Constructs a sparse matrix from the sparse expression \a other */
    template<typename OtherDerived>
    inline SparseMatrix(const SparseMatrixBase<OtherDerived>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      const bool needToTranspose = (Flags & RowMajorBit) != (internal::evaluator<OtherDerived>::Flags & RowMajorBit);
      if (needToTranspose)
        *this = other.derived();
      else
      {
        #ifdef EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
          EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
        #endif
        internal::call_assignment_no_alias(*this, other.derived());
      }
    }

    /** Constructs a sparse matrix from the sparse selfadjoint view \a other */
    template<typename OtherDerived, unsigned int UpLo>
    inline SparseMatrix(const SparseSelfAdjointView<OtherDerived, UpLo>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      Base::operator=(other);
    }

    inline SparseMatrix(SparseMatrix&& other) : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      *this = other.derived().markAsRValue();
    }

    /** Copy constructor (it performs a deep copy) */
    inline SparseMatrix(const SparseMatrix& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      *this = other.derived();
    }

    /** \brief Copy constructor with in-place evaluation */
    template<typename OtherDerived>
    SparseMatrix(const ReturnByValue<OtherDerived>& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      initAssignment(other);
      other.evalTo(*this);
    }

    /** \brief Copy constructor with in-place evaluation */
    template<typename OtherDerived>
    explicit SparseMatrix(const DiagonalBase<OtherDerived>& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      *this = other.derived();
    }

    /** Swaps the content of two sparse matrices of the same type.
      * This is a fast operation that simply swaps the underlying pointers and parameters. */
    inline void swap(SparseMatrix& other)
    {
      //EIGEN_DBG_SPARSE(std::cout << "SparseMatrix:: swap\n");
      std::swap(m_outerIndex, other.m_outerIndex);
      std::swap(m_innerSize, other.m_innerSize);
      std::swap(m_outerSize, other.m_outerSize);
      std::swap(m_innerNonZeros, other.m_innerNonZeros);
      m_data.swap(other.m_data);
    }

    /** Sets *this to the identity matrix.
      * This function also turns the matrix into compressed mode, and drop any reserved memory. */
    inline void setIdentity()
    {
      eigen_assert(rows() == cols() && "ONLY FOR SQUARED MATRICES");
      if (m_innerNonZeros) {
        internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
        m_innerNonZeros = 0;
      }
      m_data.resize(rows());
      // is it necessary to squeeze?
      m_data.squeeze();
      typename IndexVector::AlignedMapType outerIndexMap(outerIndexPtr(), rows() + 1);
      typename IndexVector::AlignedMapType innerIndexMap(innerIndexPtr(), rows());
      typename ScalarVector::AlignedMapType valueMap(valuePtr(), rows());
      outerIndexMap.setEqualSpaced(StorageIndex(0), StorageIndex(1));
      innerIndexMap.setEqualSpaced(StorageIndex(0), StorageIndex(1));
      valueMap.setOnes();
    }

    inline SparseMatrix& operator=(const SparseMatrix& other)
    {
      if (other.isRValue())
      {
        swap(other.const_cast_derived());
      }
      else if(this!=&other)
      {
        #ifdef EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
          EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
        #endif
        initAssignment(other);
        if(other.isCompressed())
        {
          internal::smart_copy(other.m_outerIndex, other.m_outerIndex + m_outerSize + 1, m_outerIndex);
          m_data = other.m_data;
        }
        else
        {
          Base::operator=(other);
        }
      }
      return *this;
    }

    inline SparseMatrix& operator=(SparseMatrix&& other) {
      return *this = other.derived().markAsRValue();
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename OtherDerived>
    inline SparseMatrix& operator=(const EigenBase<OtherDerived>& other)
    { return Base::operator=(other.derived()); }

    template<typename Lhs, typename Rhs>
    inline SparseMatrix& operator=(const Product<Lhs,Rhs,AliasFreeProduct>& other);
#endif // EIGEN_PARSED_BY_DOXYGEN

    template<typename OtherDerived>
    EIGEN_DONT_INLINE SparseMatrix& operator=(const SparseMatrixBase<OtherDerived>& other);

#ifndef EIGEN_NO_IO
    friend std::ostream & operator << (std::ostream & s, const SparseMatrix& m)
    {
      EIGEN_DBG_SPARSE(
        s << "Nonzero entries:\n";
        if(m.isCompressed())
        {
          for (Index i=0; i<m.nonZeros(); ++i)
            s << "(" << m.m_data.value(i) << "," << m.m_data.index(i) << ") ";
        }
        else
        {
          for (Index i=0; i<m.outerSize(); ++i)
          {
            Index p = m.m_outerIndex[i];
            Index pe = m.m_outerIndex[i]+m.m_innerNonZeros[i];
            Index k=p;
            for (; k<pe; ++k) {
              s << "(" << m.m_data.value(k) << "," << m.m_data.index(k) << ") ";
            }
            for (; k<m.m_outerIndex[i+1]; ++k) {
              s << "(_,_) ";
            }
          }
        }
        s << std::endl;
        s << std::endl;
        s << "Outer pointers:\n";
        for (Index i=0; i<m.outerSize(); ++i) {
          s << m.m_outerIndex[i] << " ";
        }
        s << " $" << std::endl;
        if(!m.isCompressed())
        {
          s << "Inner non zeros:\n";
          for (Index i=0; i<m.outerSize(); ++i) {
            s << m.m_innerNonZeros[i] << " ";
          }
          s << " $" << std::endl;
        }
        s << std::endl;
      );
      s << static_cast<const SparseMatrixBase<SparseMatrix>&>(m);
      return s;
    }
#endif

    /** Destructor */
    inline ~SparseMatrix()
    {
      internal::conditional_aligned_delete_auto<StorageIndex, true>(m_outerIndex, m_outerSize + 1);
      internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
    }

    /** Overloaded for performance */
    Scalar sum() const;
    
#   ifdef EIGEN_SPARSEMATRIX_PLUGIN
#     include EIGEN_SPARSEMATRIX_PLUGIN
#   endif

protected:

    template<typename Other>
    void initAssignment(const Other& other)
    {
      resize(other.rows(), other.cols());
      if(m_innerNonZeros)
      {
        internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
        m_innerNonZeros = 0;
      }
    }

    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_DEPRECATED EIGEN_DONT_INLINE Scalar& insertCompressed(Index row, Index col);

    /** \internal
      * A vector object that is equal to 0 everywhere but v at the position i */
    class SingletonVector
    {
        StorageIndex m_index;
        StorageIndex m_value;
      public:
        typedef StorageIndex value_type;
        SingletonVector(Index i, Index v)
          : m_index(convert_index(i)), m_value(convert_index(v))
        {}

        StorageIndex operator[](Index i) const { return i==m_index ? m_value : 0; }
    };

    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_DEPRECATED EIGEN_DONT_INLINE Scalar& insertUncompressed(Index row, Index col);

public:
    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_STRONG_INLINE Scalar& insertBackUncompressed(Index row, Index col)
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      eigen_assert(!isCompressed());
      eigen_assert(m_innerNonZeros[outer]<=(m_outerIndex[outer+1] - m_outerIndex[outer]));

      Index p = m_outerIndex[outer] + m_innerNonZeros[outer]++;
      m_data.index(p) = convert_index(inner);
      return (m_data.value(p) = Scalar(0));
    }
protected:
    struct IndexPosPair {
      IndexPosPair(Index a_i, Index a_p) : i(a_i), p(a_p) {}
      Index i;
      Index p;
    };

    /** \internal assign \a diagXpr to the diagonal of \c *this
      * There are different strategies:
      *   1 - if *this is overwritten (Func==assign_op) or *this is empty, then we can work treat *this as a dense vector expression.
      *   2 - otherwise, for each diagonal coeff,
      *     2.a - if it already exists, then we update it,
      *     2.b - if the correct position is at the end of the vector, and there is capacity, push to back
      *     2.b - otherwise, the insertion requires a data move, record insertion locations and handle in a second pass
      *   3 - at the end, if some entries failed to be updated in-place, then we alloc a new buffer, copy each chunk at the right position, and insert the new elements.
      */
    template<typename DiagXpr, typename Func>
    void assignDiagonal(const DiagXpr diagXpr, const Func& assignFunc)
    {
      
      constexpr StorageIndex kEmptyIndexVal(-1);
      typedef typename IndexVector::AlignedMapType IndexMap;
      typedef typename ScalarVector::AlignedMapType ValueMap;

      Index n = diagXpr.size();

      const bool overwrite = internal::is_same<Func, internal::assign_op<Scalar,Scalar> >::value;
      if(overwrite)
      {
        if((this->rows()!=n) || (this->cols()!=n))
          this->resize(n, n);
      }

      if(data().size()==0 || overwrite)
      {
        if (!isCompressed()) {
          internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
          m_innerNonZeros = 0;
        }
        resizeNonZeros(n);
        IndexMap outerIndexMap(outerIndexPtr(), n + 1);
        IndexMap innerIndexMap(innerIndexPtr(), n);
        ValueMap valueMap(valuePtr(), n);
        outerIndexMap.setEqualSpaced(StorageIndex(0), StorageIndex(1));
        innerIndexMap.setEqualSpaced(StorageIndex(0), StorageIndex(1));
        valueMap.setZero();
        internal::call_assignment_no_alias(valueMap, diagXpr, assignFunc);
      }
      else
      {
          internal::evaluator<DiagXpr> diaEval(diagXpr);

          ei_declare_aligned_stack_constructed_variable(StorageIndex, tmp, n, 0);
          typename IndexVector::AlignedMapType insertionLocations(tmp, n);
          insertionLocations.setConstant(kEmptyIndexVal);

          Index deferredInsertions = 0;
          Index shift = 0;

          for (Index j = 0; j < n; j++) {
            Index begin = outerIndexPtr()[j];
            Index end = isCompressed() ? outerIndexPtr()[j + 1] : begin + innerNonZeroPtr()[j];
            Index capacity = outerIndexPtr()[j + 1] - end;
            Index dst = data().searchLowerIndex(begin, end, j);
            // the entry exists: update it now
            if (dst != end && data().index(dst) == j) assignFunc.assignCoeff(data().value(dst), diaEval.coeff(j));
            // the entry belongs at the back of the vector: push to back
            else if (dst == end && capacity > 0)
              assignFunc.assignCoeff(insertBackUncompressed(j, j), diaEval.coeff(j));
            // the insertion requires a data move, record insertion location and handle in second pass
            else {
              insertionLocations.coeffRef(j) = StorageIndex(dst);
              deferredInsertions++;
              // if there is no capacity, all vectors to the right of this are shifted
              if (capacity == 0) shift++;
            }
          }
          
          if (deferredInsertions > 0) {

            data().resize(data().size() + shift);
            Index copyEnd = isCompressed() ? outerIndexPtr()[outerSize()]
                                           : outerIndexPtr()[outerSize() - 1] + innerNonZeroPtr()[outerSize() - 1];
            for (Index j = outerSize() - 1; deferredInsertions > 0; j--) {
              Index begin = outerIndexPtr()[j];
              Index end = isCompressed() ? outerIndexPtr()[j + 1] : begin + innerNonZeroPtr()[j];
              Index capacity = outerIndexPtr()[j + 1] - end;

              bool doInsertion = insertionLocations(j) >= 0;
              bool breakUpCopy = doInsertion && (capacity > 0);
              // break up copy for sorted insertion into inactive nonzeros
              // optionally, add another criterium, i.e. 'breakUpCopy || (capacity > threhsold)'
              // where `threshold >= 0` to skip inactive nonzeros in each vector
              // this reduces the total number of copied elements, but requires more moveChunk calls
              if (breakUpCopy) {
                Index copyBegin = outerIndexPtr()[j + 1];
                Index to = copyBegin + shift;
                Index chunkSize = copyEnd - copyBegin;
                if (chunkSize > 0) data().moveChunk(copyBegin, to, chunkSize);
                copyEnd = end;
              }

              outerIndexPtr()[j + 1] += shift;
              
              if (doInsertion) {
                // if there is capacity, shift into the inactive nonzeros
                if (capacity > 0) shift++;
                Index copyBegin = insertionLocations(j);
                Index to = copyBegin + shift;
                Index chunkSize = copyEnd - copyBegin;
                if (chunkSize > 0) data().moveChunk(copyBegin, to, chunkSize);
                Index dst = to - 1;
                data().index(dst) = StorageIndex(j);
                assignFunc.assignCoeff(data().value(dst) = Scalar(0), diaEval.coeff(j));
                if (!isCompressed()) innerNonZeroPtr()[j]++;
                shift--;
                deferredInsertions--;
                copyEnd = copyBegin;
              }
            }
          }     
          eigen_assert((shift == 0) && (deferredInsertions == 0));
      }
    }

    /* Provides a consistent reserve and reallocation strategy for insertCompressed and insertUncompressed
    If there is insufficient space for one insertion, the size of the memory buffer is doubled */
    inline void checkAllocatedSpaceAndMaybeExpand();
    /* These functions are used to avoid a redundant binary search operation in functions such as coeffRef() and assume `dst` is the appropriate sorted insertion point */
    EIGEN_STRONG_INLINE Scalar& insertAtByOuterInner(Index outer, Index inner, Index dst);
    Scalar& insertCompressedAtByOuterInner(Index outer, Index inner, Index dst);
    Scalar& insertUncompressedAtByOuterInner(Index outer, Index inner, Index dst);

private:
  EIGEN_STATIC_ASSERT(NumTraits<StorageIndex>::IsSigned,THE_INDEX_TYPE_MUST_BE_A_SIGNED_TYPE)
  EIGEN_STATIC_ASSERT((Options&(ColMajor|RowMajor))==Options,INVALID_MATRIX_TEMPLATE_PARAMETERS)

  struct default_prunning_func {
    default_prunning_func(const Scalar& ref, const RealScalar& eps) : reference(ref), epsilon(eps) {}
    inline bool operator() (const Index&, const Index&, const Scalar& value) const
    {
      return !internal::isMuchSmallerThan(value, reference, epsilon);
    }
    Scalar reference;
    RealScalar epsilon;
  };
};

namespace internal {

// Creates a compressed sparse matrix from a range of unsorted triplets
// Requires temporary storage to handle duplicate entries
template <typename InputIterator, typename SparseMatrixType, typename DupFunctor>
void set_from_triplets(const InputIterator& begin, const InputIterator& end, SparseMatrixType& mat,
                       DupFunctor dup_func) {

  constexpr bool IsRowMajor = SparseMatrixType::IsRowMajor;
  typedef typename SparseMatrixType::StorageIndex StorageIndex;
  typedef typename VectorX<StorageIndex>::AlignedMapType IndexMap;
  if (begin == end) return;

  // free innerNonZeroPtr (if present) and zero outerIndexPtr
  mat.resize(mat.rows(), mat.cols());
  // allocate temporary storage for nonzero insertion (outer size) and duplicate removal (inner size)
  ei_declare_aligned_stack_constructed_variable(StorageIndex, tmp, numext::maxi(mat.innerSize(), mat.outerSize()), 0);
  // scan triplets to determine allocation size before constructing matrix
  IndexMap outerIndexMap(mat.outerIndexPtr(), mat.outerSize() + 1);
  for (InputIterator it(begin); it != end; ++it) {
    eigen_assert(it->row() >= 0 && it->row() < mat.rows() && it->col() >= 0 && it->col() < mat.cols());
    StorageIndex j = IsRowMajor ? it->row() : it->col();
    outerIndexMap.coeffRef(j + 1)++;
  }

  // finalize outer indices and allocate memory
  std::partial_sum(outerIndexMap.begin(), outerIndexMap.end(), outerIndexMap.begin());
  Index nonZeros = mat.outerIndexPtr()[mat.outerSize()];
  mat.resizeNonZeros(nonZeros);

  // use tmp to track nonzero insertions
  IndexMap back(tmp, mat.outerSize());
  back = outerIndexMap.head(mat.outerSize());

  // push triplets to back of each inner vector
  for (InputIterator it(begin); it != end; ++it) {
    StorageIndex j = IsRowMajor ? it->row() : it->col();
    StorageIndex i = IsRowMajor ? it->col() : it->row();
    mat.data().index(back.coeff(j)) = i;
    mat.data().value(back.coeff(j)) = it->value();
    back.coeffRef(j)++;
  }

  // use tmp to collapse duplicates
  IndexMap wi(tmp, mat.innerSize());
  mat.collapseDuplicates(wi, dup_func);
  mat.sortInnerIndices();
}

// Creates a compressed sparse matrix from a sorted range of triplets
template <typename InputIterator, typename SparseMatrixType, typename DupFunctor>
void set_from_triplets_sorted(const InputIterator& begin, const InputIterator& end, SparseMatrixType& mat,
                              DupFunctor dup_func) {
  constexpr bool IsRowMajor = SparseMatrixType::IsRowMajor;
  typedef typename SparseMatrixType::StorageIndex StorageIndex;
  typedef typename VectorX<StorageIndex>::AlignedMapType IndexMap;
  if (begin == end) return;

  constexpr StorageIndex kEmptyIndexValue(-1);
  // deallocate inner nonzeros if present and zero outerIndexPtr
  mat.resize(mat.rows(), mat.cols());
  // use outer indices to count non zero entries (excluding duplicate entries)
  StorageIndex previous_j = kEmptyIndexValue;
  StorageIndex previous_i = kEmptyIndexValue;
  // scan triplets to determine allocation size before constructing matrix
  IndexMap outerIndexMap(mat.outerIndexPtr(), mat.outerSize() + 1);
  for (InputIterator it(begin); it != end; ++it) {
    eigen_assert(it->row() >= 0 && it->row() < mat.rows() && it->col() >= 0 && it->col() < mat.cols());
    StorageIndex j = IsRowMajor ? it->row() : it->col();
    StorageIndex i = IsRowMajor ? it->col() : it->row();
    eigen_assert(j > previous_j || (j == previous_j && i >= previous_i));
    // identify duplicates by examining previous location
    bool duplicate = (previous_j == j) && (previous_i == i);
    if (!duplicate) outerIndexMap.coeffRef(j + 1)++;
    previous_j = j;
    previous_i = i;
  }
  
  // finalize outer indices and allocate memory
  std::partial_sum(outerIndexMap.begin(), outerIndexMap.end(), outerIndexMap.begin());
  Index nonZeros = mat.outerIndexPtr()[mat.outerSize()];
  mat.resizeNonZeros(nonZeros);

  previous_i = kEmptyIndexValue;
  previous_j = kEmptyIndexValue;
  Index back = 0;
  for (InputIterator it(begin); it != end; ++it) {
    StorageIndex j = IsRowMajor ? it->row() : it->col();
    StorageIndex i = IsRowMajor ? it->col() : it->row();
    bool duplicate = (previous_j == j) && (previous_i == i);
    if (duplicate) {
      mat.data().value(back - 1) = dup_func(mat.data().value(back - 1), it->value());
    } else {
      // push triplets to back
      mat.data().index(back) = i;
      mat.data().value(back) = it->value();
      previous_j = j;
      previous_i = i;
      back++;
    }
  }
  // matrix is finalized
}

}


/** Fill the matrix \c *this with the list of \em triplets defined by the iterator range \a begin - \a end.
  *
  * A \em triplet is a tuple (i,j,value) defining a non-zero element.
  * The input list of triplets does not have to be sorted, and can contains duplicated elements.
  * In any case, the result is a \b sorted and \b compressed sparse matrix where the duplicates have been summed up.
  * This is a \em O(n) operation, with \em n the number of triplet elements.
  * The initial contents of \c *this is destroyed.
  * The matrix \c *this must be properly resized beforehand using the SparseMatrix(Index,Index) constructor,
  * or the resize(Index,Index) method. The sizes are not extracted from the triplet list.
  *
  * The \a InputIterators value_type must provide the following interface:
  * \code
  * Scalar value() const; // the value
  * Scalar row() const;   // the row index i
  * Scalar col() const;   // the column index j
  * \endcode
  * See for instance the Eigen::Triplet template class.
  *
  * Here is a typical usage example:
  * \code
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(estimation_of_entries);
    for(...)
    {
      // ...
      tripletList.push_back(T(i,j,v_ij));
    }
    SparseMatrixType m(rows,cols);
    m.setFromTriplets(tripletList.begin(), tripletList.end());
    // m is ready to go!
  * \endcode
  *
  * \warning The list of triplets is read multiple times (at least twice). Therefore, it is not recommended to define
  * an abstract iterator over a complex data-structure that would be expensive to evaluate. The triplets should rather
  * be explicitly stored into a std::vector for instance.
  */
template<typename Scalar, int Options_, typename StorageIndex_>
template<typename InputIterators>
void SparseMatrix<Scalar,Options_,StorageIndex_>::setFromTriplets(const InputIterators& begin, const InputIterators& end)
{
  internal::set_from_triplets<InputIterators, SparseMatrix<Scalar,Options_,StorageIndex_> >(begin, end, *this, internal::scalar_sum_op<Scalar,Scalar>());
}

/** The same as setFromTriplets but triplets are assumed to be pre-sorted. This is faster and requires less temporary storage. 
  * Two triplets `a` and `b` are appropriately ordered if:
  * \code
  * ColMajor: ((a.col() != b.col()) ? (a.col() < b.col()) : (a.row() < b.row())
  * RowMajor: ((a.row() != b.row()) ? (a.row() < b.row()) : (a.col() < b.col())
  * \endcode
  */
template<typename Scalar, int Options_, typename StorageIndex_>
template<typename InputIterators>
void SparseMatrix<Scalar, Options_, StorageIndex_>::setFromSortedTriplets(const InputIterators& begin, const InputIterators& end)
{
    internal::set_from_triplets_sorted<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_> >(begin, end, *this, internal::scalar_sum_op<Scalar, Scalar>());
}

/** The same as setFromTriplets but when duplicates are met the functor \a dup_func is applied:
  * \code
  * value = dup_func(OldValue, NewValue)
  * \endcode
  * Here is a C++11 example keeping the latest entry only:
  * \code
  * mat.setFromTriplets(triplets.begin(), triplets.end(), [] (const Scalar&,const Scalar &b) { return b; });
  * \endcode
  */
template<typename Scalar, int Options_, typename StorageIndex_>
template<typename InputIterators, typename DupFunctor>
void SparseMatrix<Scalar, Options_, StorageIndex_>::setFromTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func)
{
    internal::set_from_triplets<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>, DupFunctor>(begin, end, *this, dup_func);
}

/** The same as setFromSortedTriplets but when duplicates are met the functor \a dup_func is applied:
  * \code
  * value = dup_func(OldValue, NewValue)
  * \endcode
  * Here is a C++11 example keeping the latest entry only:
  * \code
  * mat.setFromTriplets(triplets.begin(), triplets.end(), [] (const Scalar&,const Scalar &b) { return b; });
  * \endcode
  */
template<typename Scalar, int Options_, typename StorageIndex_>
template<typename InputIterators, typename DupFunctor>
void SparseMatrix<Scalar, Options_, StorageIndex_>::setFromSortedTriplets(const InputIterators& begin, const InputIterators& end, DupFunctor dup_func)
{
    internal::set_from_triplets_sorted<InputIterators, SparseMatrix<Scalar, Options_, StorageIndex_>, DupFunctor>(begin, end, *this, dup_func);
}

/** \internal */
template<typename Scalar, int Options_, typename StorageIndex_>
template<typename Derived, typename DupFunctor>
void SparseMatrix<Scalar, Options_, StorageIndex_>::collapseDuplicates(DenseBase<Derived>& wi, DupFunctor dup_func)
{
  eigen_assert(wi.size() >= m_innerSize);
  constexpr StorageIndex kEmptyIndexValue(-1);
  wi.setConstant(kEmptyIndexValue);
  StorageIndex count = 0;
  // for each inner-vector, wi[inner_index] will hold the position of first element into the index/value buffers
  for (Index j = 0; j < m_outerSize; ++j) {
    StorageIndex start = count;
    StorageIndex oldEnd = isCompressed() ? m_outerIndex[j + 1] : m_outerIndex[j] + m_innerNonZeros[j];
    for (StorageIndex k = m_outerIndex[j]; k < oldEnd; ++k) {
      StorageIndex i = m_data.index(k);
      if (wi(i) >= start) {
        // we already meet this entry => accumulate it
        m_data.value(wi(i)) = dup_func(m_data.value(wi(i)), m_data.value(k));
      } else {
        m_data.value(count) = m_data.value(k);
        m_data.index(count) = m_data.index(k);
        wi(i) = count;
        ++count;
      }
    }
    m_outerIndex[j] = start;
  }
  m_outerIndex[m_outerSize] = count;
  // turn the matrix into compressed form
  if (m_innerNonZeros) {
    internal::conditional_aligned_delete_auto<StorageIndex, true>(m_innerNonZeros, m_outerSize);
    m_innerNonZeros = 0;
  }
  m_data.resize(m_outerIndex[m_outerSize]);
}



/** \internal */
template<typename Scalar, int Options_, typename StorageIndex_>
template<typename OtherDerived>
EIGEN_DONT_INLINE SparseMatrix<Scalar,Options_,StorageIndex_>& SparseMatrix<Scalar,Options_,StorageIndex_>::operator=(const SparseMatrixBase<OtherDerived>& other)
{
  EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename OtherDerived::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

  #ifdef EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
    EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN
  #endif
      
  const bool needToTranspose = (Flags & RowMajorBit) != (internal::evaluator<OtherDerived>::Flags & RowMajorBit);
  if (needToTranspose)
  {
    #ifdef EIGEN_SPARSE_TRANSPOSED_COPY_PLUGIN
      EIGEN_SPARSE_TRANSPOSED_COPY_PLUGIN
    #endif
    // two passes algorithm:
    //  1 - compute the number of coeffs per dest inner vector
    //  2 - do the actual copy/eval
    // Since each coeff of the rhs has to be evaluated twice, let's evaluate it if needed
    typedef typename internal::nested_eval<OtherDerived,2,typename internal::plain_matrix_type<OtherDerived>::type >::type OtherCopy;
    typedef internal::remove_all_t<OtherCopy> OtherCopy_;
    typedef internal::evaluator<OtherCopy_> OtherCopyEval;
    OtherCopy otherCopy(other.derived());
    OtherCopyEval otherCopyEval(otherCopy);

    SparseMatrix dest(other.rows(),other.cols());
    Eigen::Map<IndexVector> (dest.m_outerIndex,dest.outerSize()).setZero();

    // pass 1
    // FIXME the above copy could be merged with that pass
    for (Index j=0; j<otherCopy.outerSize(); ++j)
      for (typename OtherCopyEval::InnerIterator it(otherCopyEval, j); it; ++it)
        ++dest.m_outerIndex[it.index()];

    // prefix sum
    StorageIndex count = 0;
    IndexVector positions(dest.outerSize());
    for (Index j=0; j<dest.outerSize(); ++j)
    {
      StorageIndex tmp = dest.m_outerIndex[j];
      dest.m_outerIndex[j] = count;
      positions[j] = count;
      count += tmp;
    }
    dest.m_outerIndex[dest.outerSize()] = count;
    // alloc
    dest.m_data.resize(count);
    // pass 2
    for (StorageIndex j=0; j<otherCopy.outerSize(); ++j)
    {
      for (typename OtherCopyEval::InnerIterator it(otherCopyEval, j); it; ++it)
      {
        Index pos = positions[it.index()]++;
        dest.m_data.index(pos) = j;
        dest.m_data.value(pos) = it.value();
      }
    }
    this->swap(dest);
    return *this;
  }
  else
  {
    if(other.isRValue())
    {
      initAssignment(other.derived());
    }
    // there is no special optimization
    return Base::operator=(other.derived());
  }
}

template <typename Scalar_, int Options_, typename StorageIndex_>
inline typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar& SparseMatrix<Scalar_, Options_, StorageIndex_>::insert(
    Index row, Index col) {
  Index outer = IsRowMajor ? row : col;
  Index inner = IsRowMajor ? col : row;
  Index start = outerIndexPtr()[outer];
  Index end = isCompressed() ? outerIndexPtr()[outer + 1] : start + innerNonZeroPtr()[outer];
  Index dst = data().searchLowerIndex(start, end, inner);
  eigen_assert((dst == end || data().index(dst) != inner) &&
      "you cannot insert an element that already exists, you must call coeffRef to this end");
  return insertAtByOuterInner(outer, inner, dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
EIGEN_STRONG_INLINE typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertAtByOuterInner(Index outer, Index inner, Index dst) {
  uncompress();
  return insertUncompressedAtByOuterInner(outer, inner, dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
EIGEN_DEPRECATED EIGEN_DONT_INLINE typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertUncompressed(Index row, Index col) {
  eigen_assert(!isCompressed());
  Index outer = IsRowMajor ? row : col;
  Index inner = IsRowMajor ? col : row;
  Index start = outerIndexPtr()[outer];
  Index end = start + innerNonZeroPtr()[outer];
  Index dst = data().searchLowerIndex(start, end, inner);
  eigen_assert((dst == end || data().index(dst) != inner) &&
               "you cannot insert an element that already exists, you must call coeffRef to this end");
  return insertUncompressedAtByOuterInner(outer, inner, dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
EIGEN_DEPRECATED EIGEN_DONT_INLINE typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertCompressed(Index row, Index col) {
  eigen_assert(isCompressed());
  Index outer = IsRowMajor ? row : col;
  Index inner = IsRowMajor ? col : row;
  Index start = outerIndexPtr()[outer];
  Index end = outerIndexPtr()[outer + 1];
  Index dst = data().searchLowerIndex(start, end, inner);
  eigen_assert((dst == end || data().index(dst) != inner) &&
               "you cannot insert an element that already exists, you must call coeffRef to this end");
  return insertCompressedAtByOuterInner(outer, inner, dst);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
EIGEN_STRONG_INLINE void SparseMatrix<Scalar_, Options_, StorageIndex_>::checkAllocatedSpaceAndMaybeExpand() {
  // if there is no capacity for a single insertion, double the capacity
  if (data().allocatedSize() <= data().size()) {
    // increase capacity by a mininum of 32
    Index minReserve = 32;
    Index reserveSize = numext::maxi(minReserve, data().allocatedSize());
    data().reserve(reserveSize);
  }
}

template <typename Scalar_, int Options_, typename StorageIndex_>
typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertCompressedAtByOuterInner(Index outer, Index inner, Index dst) {
  eigen_assert(isCompressed());
  // compressed insertion always requires expanding the buffer
  checkAllocatedSpaceAndMaybeExpand();
  data().resize(data().size() + 1);
  Index chunkSize = outerIndexPtr()[outerSize()] - dst;
  // shift the existing data to the right if necessary
  if (chunkSize > 0) data().moveChunk(dst, dst + 1, chunkSize);
  // update nonzero counts
  typename IndexVector::AlignedMapType outerIndexMap(outerIndexPtr(), outerSize() + 1);
  outerIndexMap.segment(outer + 1, outerSize() - outer).array() += 1;
  // initialize the coefficient
  data().index(dst) = StorageIndex(inner);
  // return a reference to the coefficient
  return data().value(dst) = Scalar(0);
}

template <typename Scalar_, int Options_, typename StorageIndex_>
typename SparseMatrix<Scalar_, Options_, StorageIndex_>::Scalar&
SparseMatrix<Scalar_, Options_, StorageIndex_>::insertUncompressedAtByOuterInner(Index outer, Index inner, Index dst) {
  eigen_assert(!isCompressed());
  // find nearest outer vector to the right with capacity (if any) to minimize copy size
  Index target = outer;
  for (; target < outerSize(); target++) {
    Index start = outerIndexPtr()[target];
    Index end = start + innerNonZeroPtr()[target];
    Index capacity = outerIndexPtr()[target + 1] - end;
    if (capacity > 0) {
      // `target` has room for interior insertion
      Index chunkSize = end - dst;
      // shift the existing data to the right if necessary
      if (chunkSize > 0) data().moveChunk(dst, dst + 1, chunkSize);
      break;
    }
  }
  if (target == outerSize()) {
    // no room for interior insertion (to the right of `outer`)
    target = outer;
    Index dst_offset = dst - outerIndexPtr()[target];
    Index totalCapacity = data().allocatedSize() - data().size();
    eigen_assert(totalCapacity >= 0);
    if (totalCapacity == 0) {
      // there is no room left. we must reallocate. reserve space in each vector
      constexpr StorageIndex kReserveSizePerVector(2);
      reserveInnerVectors(IndexVector::Constant(outerSize(), kReserveSizePerVector));
    } else {
      // dont reallocate, but re-distribute the remaining capacity to the right of `outer`
      // each vector in the range [outer,outerSize) will receive totalCapacity / (outerSize - outer) nonzero
      // reservations each vector in the range [outer,remainder) will receive an additional nonzero reservation where
      // remainder = totalCapacity % (outerSize - outer)
      typedef internal::sparse_reserve_op<StorageIndex> ReserveSizesOp;
      typedef CwiseNullaryOp<ReserveSizesOp, IndexVector> ReserveSizesXpr;
      ReserveSizesXpr reserveSizesXpr(outerSize(), 1, ReserveSizesOp(target, outerSize(), totalCapacity));
      eigen_assert(reserveSizesXpr.sum() == totalCapacity);
      reserveInnerVectors(reserveSizesXpr);
    }
    Index start = outerIndexPtr()[target];
    Index end = start + innerNonZeroPtr()[target];
    dst = start + dst_offset;
    Index chunkSize = end - dst;
    if (chunkSize > 0) data().moveChunk(dst, dst + 1, chunkSize);
  }
  // update nonzero counts
  innerNonZeroPtr()[outer]++;
  typename IndexVector::AlignedMapType outerIndexMap(outerIndexPtr(), outerSize() + 1);
  outerIndexMap.segment(outer + 1, target - outer).array() += 1;
  // initialize the coefficient
  data().index(dst) = StorageIndex(inner);
  // return a reference to the coefficient
  return data().value(dst) = Scalar(0);
}

namespace internal {

    template <typename Scalar_, int Options_, typename StorageIndex_>
    struct evaluator<SparseMatrix<Scalar_, Options_, StorageIndex_>>
        : evaluator<SparseCompressedBase<SparseMatrix<Scalar_, Options_, StorageIndex_>>> {
      typedef evaluator<SparseCompressedBase<SparseMatrix<Scalar_, Options_, StorageIndex_>>> Base;
      typedef SparseMatrix<Scalar_, Options_, StorageIndex_> SparseMatrixType;
      evaluator() : Base() {}
      explicit evaluator(const SparseMatrixType& mat) : Base(mat) {}
    };

}

// Specialization for SparseMatrix.
// Serializes [rows, cols, isCompressed, outerSize, innerBufferSize,
// innerNonZeros, outerIndices, innerIndices, values].
template <typename Scalar, int Options, typename StorageIndex>
class Serializer<SparseMatrix<Scalar, Options, StorageIndex>, void> {
 public:
  typedef SparseMatrix<Scalar, Options, StorageIndex> SparseMat;

  struct Header {
    typename SparseMat::Index rows;
    typename SparseMat::Index cols;
    bool compressed;
    Index outer_size;
    Index inner_buffer_size;
  };

  EIGEN_DEVICE_FUNC size_t size(const SparseMat& value) const {
    // innerNonZeros.
    std::size_t num_storage_indices = value.isCompressed() ? 0 : value.outerSize();
    // Outer indices.
    num_storage_indices += value.outerSize() + 1;
    // Inner indices.
    const StorageIndex inner_buffer_size = value.outerIndexPtr()[value.outerSize()];
    num_storage_indices += inner_buffer_size;
    // Values.
    std::size_t num_values = inner_buffer_size;
    return sizeof(Header) + sizeof(Scalar) * num_values +
           sizeof(StorageIndex) * num_storage_indices;
  }

  EIGEN_DEVICE_FUNC uint8_t* serialize(uint8_t* dest, uint8_t* end,
                                       const SparseMat& value) {
    if (EIGEN_PREDICT_FALSE(dest == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(dest + size(value) > end)) return nullptr;

    const size_t header_bytes = sizeof(Header);
    Header header = {value.rows(), value.cols(), value.isCompressed(),
                     value.outerSize(), value.outerIndexPtr()[value.outerSize()]};
    EIGEN_USING_STD(memcpy)
    memcpy(dest, &header, header_bytes);
    dest += header_bytes;

    // innerNonZeros.
    if (!header.compressed) {
      std::size_t data_bytes = sizeof(StorageIndex) * header.outer_size;
      memcpy(dest, value.innerNonZeroPtr(), data_bytes);
      dest += data_bytes;
    }

    // Outer indices.
    std::size_t data_bytes = sizeof(StorageIndex) * (header.outer_size + 1);
    memcpy(dest, value.outerIndexPtr(), data_bytes);
    dest += data_bytes;

    // Inner indices.
    data_bytes = sizeof(StorageIndex) * header.inner_buffer_size;
    memcpy(dest, value.innerIndexPtr(), data_bytes);
    dest += data_bytes;

    // Values.
    data_bytes = sizeof(Scalar) * header.inner_buffer_size;
    memcpy(dest, value.valuePtr(), data_bytes);
    dest += data_bytes;

    return dest;
  }

  EIGEN_DEVICE_FUNC const uint8_t* deserialize(const uint8_t* src,
                                               const uint8_t* end,
                                               SparseMat& value) const {
    if (EIGEN_PREDICT_FALSE(src == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(src + sizeof(Header) > end)) return nullptr;

    const size_t header_bytes = sizeof(Header);
    Header header;
    EIGEN_USING_STD(memcpy)
    memcpy(&header, src, header_bytes);
    src += header_bytes;

    value.setZero();
    value.resize(header.rows, header.cols);
    if (header.compressed) {
      value.makeCompressed();
    } else {
      value.uncompress();
    }
    
    // Adjust value ptr size.
    value.data().resize(header.inner_buffer_size);

    // Initialize compressed state and inner non-zeros.
    if (!header.compressed) {           
      // Inner non-zero counts.
      std::size_t data_bytes = sizeof(StorageIndex) * header.outer_size;
      if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
      memcpy(value.innerNonZeroPtr(), src, data_bytes);
      src += data_bytes;
    }

    // Outer indices.
    std::size_t data_bytes = sizeof(StorageIndex) * (header.outer_size + 1);
    if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
    memcpy(value.outerIndexPtr(), src, data_bytes);
    src += data_bytes;

    // Inner indices.
    data_bytes = sizeof(StorageIndex) * header.inner_buffer_size;
    if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
    memcpy(value.innerIndexPtr(), src, data_bytes);
    src += data_bytes;

    // Values.
    data_bytes = sizeof(Scalar) * header.inner_buffer_size;
    if (EIGEN_PREDICT_FALSE(src + data_bytes > end)) return nullptr;
    memcpy(value.valuePtr(), src, data_bytes);
    src += data_bytes;
    return src;
  }
};

} // end namespace Eigen

#endif // EIGEN_SPARSEMATRIX_H
