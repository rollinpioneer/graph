#include <cmath>
#include <cstdint>
extern "C" int64_t masks(const double* p,int64_t n,double w,double h,double d,int64_t* counts,int64_t* ids){
 int64_t ni=0;for(int64_t i=0;i<n;++i){double x=p[3*i],y=p[3*i+1],z=p[3*i+2];bool ht=std::abs(z)<h/2.,f=(x>d-.06)&&(x<d),side=std::abs(y)<w/2.+.01;
 counts[0]+=ht&&f&&(y>-w/2.-.01)&&(y<-w/2.);
 counts[1]+=ht&&f&&(y>w/2.)&&(y<w/2.+.01);
 counts[2]+=ht&&side&&(x<=d-.06)&&(x>d-.07);
 counts[3]+=ht&&side&&(x<=d-.07)&&(x>d-.12);
 if(ht&&f&&(std::abs(y)<=w/2.)){counts[4]++;ids[ni++]=i;}
 }return ni;
}
