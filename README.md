# assignments of cs231n

## assignment1(DONE)

官方features.py中的坑：
> 94：image = np.at_least_2d(im)

应为 image = np.atleast_2d(im)
> 121：orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T

应为 orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx//2::cx, cy//2::cy].T
(Python2与Python3除法不同)

## assignment2(DONE)

layers.py中的坑：
>在spatial batch normalization 中：最后dgamma和dbeta的返回值要keepdims而不是返回shape=（C,)

Q5.我只选择了Tensorflow

## assignment3(TODO)
