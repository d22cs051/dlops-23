# How to use:

steps:-

*Note: Make sure you are in the correct directory*

0. use ```pwd``` and ```ls``` to verify files.
1. build docker file using:
``` 
docker build . -t ass_1_docker
```
2. run the build image
```
docker run -it ass_1_docker
```
3. you can see the entire traing of the model as specifed arch.

*Note: Model is not saved to local drive*