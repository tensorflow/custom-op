// kernel_example.h
#ifndef KERNEL_TIME_TWO_H_
#define KERNEL_TIME_TWO_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct TimeTwoFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_TIME_TWO_H_
