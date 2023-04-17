# ocaml-torch-leaktest
simple example demonstrating ocaml-torch memory leak. 

Runs a MSE calculation, which compares one image to a database of images.
Python equivalent is
```
dbf = th.randn(image_count, image_res, image_res)
for i in range(30): 
	a = th.randn(image_res, image_res)
	d = th.sum((dbf - a)**2, (1,2))
	mindex = th.argmin(d)
	dist = d[mindex]
```

Two ways of performing the calculation are provided; both leak memory (to different degrees). 

An example run reports has a database size of 29 MB ( 8k 30x30 images), and leaks 1192 MB of memory over 30 runs (950 MB default libtorch -> 2142 MB). 
