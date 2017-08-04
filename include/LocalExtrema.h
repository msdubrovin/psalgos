#ifndef PSALGOS_LOCALEXTREMA_H
#define PSALGOS_LOCALEXTREMA_H
//--------------------------------------------------------------------------
// $Id: LocalExtrema.h 12925 2017-08-04 10:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description: see documentation below
//------------------------------------------------------------------------

#include <string>
#include <vector>
#include <iostream> // for cout, ostream
#include <cstddef>  // for size_t
#include <cstring>  // for memcpy
#include <cmath>    // for sqrt

#include "psalgos/Types.h"

//-----------------------------

using namespace std;

//-----------------------------

namespace localextrema {

/**
 *  @ingroup psalgos
 *
 *  @brief LocalExtrema - methods for 2-d image processing algorithms.
 *
 *  This software was developed for the LCLS project.  
 *  If you use all or part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id:$
 *
 *  @author Mikhail Dubrovin
 *
 *  @see ImgAlgos.ImgImgProc
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 *
 * 
 *  @li  Includes and typedefs
 *  @code
 *  #include <cstddef>  // for size_t
 *  #include "psalgos/LocalExtrema.h"
 *  #include "psalgos/Types.h"
 *
 *  typedef types::mask_t     mask_t;
 *  typedef types::extrim_t   extrim_t;
 *  @endcode
 *
 *
 *  @li Define input parameters
 *  \n
 *  @code
 *    const T *data = ...
 *    const mask_t *mask = ...
 *    const size_t rows = 1000;
 *    const size_t cols = 1000;
 *    const size_t rank = 5;
 *    extrim_t *map = ...
 *  @endcode
 *
 *
 *  @li Call methods
 *  \n
 *  @code
 *  mapOfLocalMinimums(data, mask, rows, cols, rank, map);
 *  mapOfLocalMaximums(data, mask, rows, cols, rank, map);
 *  mapOfLocalMaximumsRank1Cross(data, mask, rows, cols, map);

 *  std::vector<TwoIndexes> v = evaluateDiagIndexes(const size_t& rank);
 *  printMatrixOfDiagIndexes(rank);
 *  printVectorOfDiagIndexes(rank);
 *  
 *  @endcode
 */

//-----------------------------

typedef types::mask_t     mask_t;
typedef types::extrim_t   extrim_t;
typedef types::TwoIndexes TwoIndexes;

//-----------------------------

std::vector<TwoIndexes> evaluateDiagIndexes(const size_t& rank);
void printMatrixOfDiagIndexes(const size_t& rank);
void printVectorOfDiagIndexes(const size_t& rank);

//-----------------------------
  /**
   * @brief returns map of local minimums of requested rank, 
   *        where rank defins a square region around central pixel [rowc-rank, rowc+rank], [colc-rank, colc+rank].
   * 
   * Map of local minumum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2/+4 for non-minumum / minumum in column / minumum in row / minimum in square of radius rank.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] map  - pointer to map of local minimums
   */
//-----------------------------

template <typename T>
void mapOfLocalMinimums( const T *data
                       , const mask_t *mask
                       , const size_t& rows
                       , const size_t& cols
                       , const size_t& rank
                       , extrim_t *map
                       )
{
  // MsgLog(_name(), debug, "in napOfLocalMinimums, rank=" << rank << "\n);

  // initialization of indexes
  //if(v_inddiag.empty())   
  std::vector<TwoIndexes> v_inddiag = evaluateDiagIndexes(rank);

  extrim_t *m_local_minimums = map;

  //if(m_local_minimums.empty()) 
  //   m_local_minimums = make_ndarray<extrim_t>(data.shape()[0], data.shape()[1]);
  std::fill_n(&m_local_minimums[0], int(rows*cols), extrim_t(0));

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;
  int irank = (int)rank;

  int irc=0;
  int ircd=0;
  int irdc=0;
  int iric=0;

  // check rank minimum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;

      if(!mask[irc]) continue;
      m_local_minimums[irc] = 1;

      // positive side of c 
      unsigned dmax = min((int)cmax-1, int(c)+irank);
      for(unsigned cd=c+1; cd<=dmax; cd++) {
        ircd = r*cols+cd;
	if(mask[ircd] && (data[ircd] < data[irc])) { 
          m_local_minimums[irc] &=~1; // clear 1st bit
          c=cd-1; // jump ahead 
	  break;
	}
      }

      if(m_local_minimums[irc] & 1) {
        // negative side of c 
        unsigned dmin = max((int)cmin, int(c)-irank);
        for(unsigned cd=dmin; cd<c; cd++) {
          ircd = r*cols+cd;
	  if(mask[ircd] && (data[ircd] < data[irc])) { 
            m_local_minimums[irc] &=~1; // clear 1st bit
            c=cd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(m_local_minimums[irc] & 1) c+=rank;
    }
  }

  // check rank minimum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {
    for(unsigned r = rmin; r<rmax; r++) {
      // if it is not a local maximum from previous algorithm
      //if(!m_local_minimums[irc]) continue;
      irc = r*cols+c;

      if(!mask[irc]) continue;
      m_local_minimums[irc] |= 2; // set 2nd bit

      // positive side of r 
      unsigned dmax = min((int)rmax-1, int(r)+irank);
      for(unsigned rd=r+1; rd<=dmax; rd++) {
        irdc = rd*cols+c;
	if(mask[irdc] && (data[irdc] < data[irc])) { 
          m_local_minimums[irc] &=~2; // clear 2nd bit
          r=rd-1; // jump ahead 
	  break;
	}
      }

      if(m_local_minimums[irc] & 2) {
        // negative side of r
        unsigned dmin = max((int)rmin, int(r)-irank);
        for(unsigned rd=dmin; rd<r; rd++) {
          irdc = rd*cols+c;
	  if(mask[irdc] && (data[irdc] < data[irc])) { 
            m_local_minimums[irc] &=~2; // clear 2nd bit
            r=rd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(m_local_minimums[irc] & 2) r+=rank;
    }
  }

  // check rank minimum in "diagonal" regions and set the 3rd bit (4)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;
      // if it is not a local minimum from two previous algorithm
      if(m_local_minimums[irc] != 3) continue;
      m_local_minimums[irc] |= 4; // set 3rd bit

      for(vector<TwoIndexes>::const_iterator ij  = v_inddiag.begin();
                                             ij != v_inddiag.end(); ij++) {
        int ir = r + (ij->i);
        int ic = c + (ij->j);

        if(  ir<(int)rmin)  continue;
        if(  ic<(int)cmin)  continue;
        if(!(ir<(int)rmax)) continue;
        if(!(ic<(int)cmax)) continue;

        iric = ir*cols+ic;
	if(mask[iric] && (data[iric] < data[irc])) {
          m_local_minimums[irc] &=~4; // clear 3rd bit
	  break;
	}
      }

      // (r,c) is a local peak, jump ahead through the tested rank range
      if(m_local_minimums[irc] & 4) c+=rank;
    }
  }
}

//--------------------
  /**
   * @brief returns map of local maximums of requested rank, 
   *        where rank defins a square region around central pixel [rowc-rank, rowc+rank], [colc-rank, colc+rank].
   * 
   * Map of local maximum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2/+4 for non-maximum / maximum in column / maximum in row / minimum in square of radius rank.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[in]  rank - radius of the square region in which central pixel has a maximal value
   * @param[out] map  - pointer to map of local maximums
   */

template <typename T>
void mapOfLocalMaximums( const T *data
                       , const mask_t *mask
                       , const size_t& rows
                       , const size_t& cols
                       , const size_t& rank
                       , extrim_t *map
                       )
{
  //MsgLog(_name(), debug, "in mapOfLocalMaximums, rank=" << rank << "\n");

  // initialization of indexes
  std::vector<TwoIndexes> v_inddiag = evaluateDiagIndexes(rank);

  extrim_t *m_local_maximums = map;
  std::fill_n(&m_local_maximums[0], int(rows*cols), extrim_t(0));

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;
  int irank = (int)rank;

  int irc=0;
  int ircd=0;
  int irdc=0;
  int iric=0;

  // check rank maximum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      irc = r*cols+c;
      if(!mask[irc]) continue;
      m_local_maximums[irc] = 1;

      // positive side of c 
      unsigned dmax = min((int)cmax-1, int(c)+irank);
      for(unsigned cd=c+1; cd<=dmax; cd++) {
        ircd = r*cols+cd;
	if(mask[ircd] && (data[ircd] > data[irc])) { 
          m_local_maximums[irc] &=~1; // clear 1st bit
          c=cd-1; // jump ahead 
	  break;
	}
      }

      if(m_local_maximums[irc] & 1) {
        // negative side of c 
        unsigned dmin = max((int)cmin, int(c)-irank);
        for(unsigned cd=dmin; cd<c; cd++) {
          ircd = r*cols+cd;
	  if(mask[ircd] && (data[ircd] > data[irc])) { 
            m_local_maximums[irc] &=~1; // clear 1st bit
            c=cd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(m_local_maximums[irc] & 1) c+=rank;
    }
  }

  // check rank maximum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {
    for(unsigned r = rmin; r<rmax; r++) {
      irc = r*cols+c;
      // if it is not a local maximum from previous algorithm
      //if(!m_local_maximums[irc]) continue;

      if(!mask[irc]) continue;
      m_local_maximums[irc] |= 2; // set 2nd bit

      // positive side of r 
      unsigned dmax = min((int)rmax-1, int(r)+irank);
      for(unsigned rd=r+1; rd<=dmax; rd++) {
        irdc = rd*cols+c;
	if(mask[irdc] && (data[irdc] > data[irc])) { 
          m_local_maximums[irc] &=~2; // clear 2nd bit
          r=rd-1; // jump ahead 
	  break;
	}
      }

      if(m_local_maximums[irc] & 2) {
        // negative side of r
        unsigned dmin = max((int)rmin, int(r)-irank);
        for(unsigned rd=dmin; rd<r; rd++) {
          irdc = rd*cols+c;
	  if(mask[irdc] && (data[irdc] > data[irc])) { 
            m_local_maximums[irc] &=~2; // clear 2nd bit
            r=rd+rank; // jump ahead 
	    break;
	  }
        }
      }

      // (r,c) is a local dip, jump ahead through the tested rank range
      if(m_local_maximums[irc] & 2) r+=rank;
    }
  }

  // check rank maximum in "diagonal" regions and set the 3rd bit (4)
  for(unsigned r = rmin; r<rmax; r++) {
    for(unsigned c = cmin; c<cmax; c++) {
      // if it is not a local maximum from two previous algorithm
      irc = r*cols+c;

      if(m_local_maximums[irc] != 3) continue;
      m_local_maximums[irc] |= 4; // set 3rd bit

      for(vector<TwoIndexes>::const_iterator ij  = v_inddiag.begin();
                                             ij != v_inddiag.end(); ij++) {
        int ir = r + (ij->i);
        int ic = c + (ij->j);

        if(  ir<(int)rmin)  continue;
        if(  ic<(int)cmin)  continue;
        if(!(ir<(int)rmax)) continue;
        if(!(ic<(int)cmax)) continue;

        iric = ir*cols+ic;
	if(mask[iric] && (data[iric] > data[irc])) {
          m_local_maximums[irc] &=~4; // clear 3rd bit
	  break;
	}
      }

      // (r,c) is a local peak, jump ahead through the tested rank range
      if(m_local_maximums[irc] & 4) c+=rank;
    }
  }
}

//-----------------------------

  /**
   * @brief returns map of local maximums of runk=1 cross(+) region (very special case for Chuck's algorithm).
   * 
   * Map of local maximum is a 2-d array of (uint16) values of data shape, 
   * with 0/+1/+2 for non-maximum / maximum in column / maximum in row, then local maximum in cross = 3.   
   * @param[in]  data - pointer to data array
   * @param[in]  mask - pointer to mask array; mask marks bad/good (0/1) pixels
   * @param[in]  rows - number of rows in all 2-d arrays
   * @param[in]  cols - number of columns in all 2-d arrays
   * @param[out] map  - pointer to map of local maximums
   */

template <typename T>
void
mapOfLocalMaximumsRank1Cross( const T *data
                            , const mask_t *mask
                            , const size_t& rows
                            , const size_t& cols
                            , extrim_t *map
                            )
{
  //MsgLog(_name(), debug, "in mapOfLocalMaximumsRank1Cross");

  extrim_t *m_local_maximums = map;
  std::fill_n(&m_local_maximums[0], int(rows*cols), extrim_t(0));

  unsigned rmin = 0;
  unsigned rmax = (int)rows;
  unsigned cmin = 0;
  unsigned cmax = (int)cols;

  int irc=0;
  int ircm=0;
  int ircm2=0;

  // check local maximum in columns and set the 1st bit (1)
  for(unsigned r = rmin; r<rmax; r++) {

    // first pixel in the row
    unsigned c = cmin;
    irc = r*cols+c;
    if(mask[irc] && mask[irc+1] && (data[irc] > data[irc+1])) {
      m_local_maximums[irc] |= 1;  // set 1st bit
      c+=2;
    }
    else c+=1;

    // all internal pixels in the row
    for(; c<cmax-1; c++) {
      irc = r*cols+c;
      if(!mask[irc]) continue;                                       // go to the next pixel
      if(mask[irc+1] && (data[irc+1] > data[irc])) continue;         // go to the next pixel
      if(mask[irc-1] && (data[irc-1] > data[irc])) {c+=1; continue;} // jump ahead 
      m_local_maximums[irc] |= 1;  // set 1st bit
      c+=1; // jump ahead 
    }

    // last pixel in the row
    ircm = r*cols+cmax-1;
    if(mask[ircm] && mask[ircm-1] && (data[ircm] > data[ircm-1])) m_local_maximums[ircm] |= 1;  // set 1st bit
  } // rows loop

  // check local maximum in rows and set the 2nd bit (2)
  for(unsigned c = cmin; c<cmax; c++) {

    // first pixel in the column
    unsigned r = rmin;
    irc  = r*cols+c;
    ircm = (r+1)*cols+c;
    if(mask[irc] && mask[ircm] && (data[irc] > data[ircm])) {
      m_local_maximums[irc] |= 2; // set 2nd bit
      r+=2;
    }
    else r+=1;

    // all internal pixels in the column
    for(; r<rmax-1; r++) {
      irc = r*cols+c;
      if(!mask[irc]) continue;
      ircm = (r+1)*cols+c;
      if(mask[ircm] && (data[ircm] > data[irc])) continue;         // go to the next pixel
      ircm = (r-1)*cols+c;
      if(mask[ircm] && (data[ircm] > data[irc])) {r+=1; continue;} // jump ahead 
      m_local_maximums[irc] |= 2; // set 2nd bit
      r+=1; // jump ahead 
    }

    // last pixel in the column
    ircm  = (rmax-1)*cols+c;
    ircm2 = (rmax-2)*cols+c;
    if(mask[ircm] && mask[ircm2] && (data[ircm] > data[ircm2])) m_local_maximums[ircm] |= 2;  // set 2nd bit
  } // columns loop
}

//-----------------------------

//-----------------------------

//-----------------------------

//-----------------------------
} // namespace localextrema
//-----------------------------
#endif // PSALGOS_LOCALEXTREMA_H
//-----------------------------
