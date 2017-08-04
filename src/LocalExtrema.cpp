
//-----------------------------

//#include "psalgos/Types.h" // TwoIndexes
#include "psalgos/LocalExtrema.h"
//#include <cmath>     // floor, ceil
//#include <iomanip>   // for std::setw
#include <sstream>   // for stringstream

//-----------------------------

//typedef psalgos::types::TwoIndexes TwoIndexes;
//typedef psalgos::localextrema::TwoIndexes TwoIndexes;

using namespace std;

//-----------------------------

namespace localextrema {

//-----------------------------

std::vector<TwoIndexes> evaluateDiagIndexes(const size_t& rank)
{
  //MsgLog(_name(), debug, "in evaluateDiagIndexes, rank=" << rank);

  std::vector<TwoIndexes> v_inddiag;

  int indmax =  rank;
  int indmin = -rank;
  unsigned npixmax = (2*rank+1)*(2*rank+1);
  if(v_inddiag.capacity() < npixmax) v_inddiag.reserve(npixmax);
  v_inddiag.clear();

  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {

      // use rectangular region of radius = rank
      // remove already tested central row and column
      if (i==0 || j==0) continue;
      // use ring region (if un-commented)
      //if (m_rank>2 && floor(std::sqrt(float(i*i + j*j)))>(int)m_rank) continue;
      //TwoIndexes inds = {i,j};
      TwoIndexes inds(i,j);
      v_inddiag.push_back(inds);
    }
  }

  //if(m_pbits & 2) printMatrixOfDiagIndexes();

  return v_inddiag;
}

//-----------------------------

void printMatrixOfDiagIndexes(const size_t& rank)
{
  int indmax = rank;
  int indmin =-rank;

  std::stringstream ss; 
  ss << "In printMatrixOfDiagIndexes, rank=" << rank << '\n';

  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {
      int status = 1;
      if (i==0 || j==0) status = 0;
      //if (m_rank>2 && floor(std::sqrt(float(i*i + j*j)))>(int)m_rank) status = 0;
      if (i==0 && j==0) ss << " +";
      else              ss << " " << status;
    }
    ss << '\n';
  }

  //MsgLog(_name(), info, ss.str());
  cout << ss.str();
}

//-----------------------------

void printVectorOfDiagIndexes(const size_t& rank)
{
  vector<TwoIndexes> v_inddiag = evaluateDiagIndexes(rank);

  std::stringstream ss; 
  ss << "In printVectorOfDiagIndexes:\n Vector size: " << v_inddiag.size() << '\n';
  int n_pairs_in_line=0;
  for( vector<TwoIndexes>::const_iterator ij  = v_inddiag.begin();
                                          ij != v_inddiag.end(); ij++ ) {
    ss << " (" << ij->i << "," << ij->j << ")";
    if ( ++n_pairs_in_line > 9 ) {ss << "\n"; n_pairs_in_line=0;}
  }   

  //MsgLog(_name(), info, ss.str());
  cout << ss.str();
}


//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------

} // namespace localextrema

//-----------------------------
