# How to calculate the transformation matrix between NED and OpenCV coordinate systems

## NED (North-East-Down) Coordinate System

- A north-east-down (NED) system uses the Cartesian coordinates (xNorth, yEast, zDown) to represent position relative to a local origin. 

- In NED, we have the x-axis points forward, the y-axis to the right, and the z-axis downward.

```plain
                  / +x (to forward, North)
                /
              /
 (Origin O) /_ _ _ _ _ _ _ _ +y (to right, East)
            |
            |
            |
            | +z (Down)

    NED Coordinate, Right-hand Coordinate System,
    assuming your eye is behind the y-O-z plane and seeing +x forward.
```

## OpenCV Coordinate System

- OpenCV coordinate system uses the Cartesian coordinates as the x-axis pointing to the right, the y-axis downward, and the z-axis forward.

```plain
                  / +z (to Forward)
                /
              /
 (Origin O) /_ _ _ _ _ _ _   +x (to Right)
            |
            |
            |
            | +y (Down)

    OpenCV Coordinate, Right-hand Coordinate System,
    assuming your eye is behind the x-O-y plane and seeing +z forward. 
```

## Notation

Assume we have the following coordinate systems:

- `wned`: the world coordinate in NED (x Forward, y Right, z Down) format;
- `cned`: the camera coordinate in NED (x Forward, y Right, z Down) format;
- `w`: the world coordinate in OpenCV style (x Right, y Down, z Forward);
- `c`: the camera coordinate in OpenCV style (x Right, y Down, z Forward);

## Why We Need OpenCV-style Camera Pose

It is because we use the following pipeline to connect RGB, camera, and world:


RGB image $(x,y)$ with $x$ pointing to the right, $y$ down, and image `origin` in the `left-top corner`
---> camera intrinsic K and inverse invK ---> camera points $P^{c}$ = $(X^{c}, Y^{c},Z^{c})$
---> camera extrinsic E and inverse invE ---> world points $P^{w}$ = $(X^{w}, Y^{w},Z^{w})$.


##  How to get the transformation matrix from NED to OpenCV Style

- The matrix is defined as $T^{w}_{wned}$ to map the points $P^{wned}$ to the points $P^{w}$, i.e., 

$P^{w}$ = $T^{w}_{wned}$ * $P^{wned}$

- The matrix is `also` defined as $T^{c}_{cned}$ to map the points $P^{cned}$ to the points $P^{c}$, i.e., 

$P^{c}$ = $T^{c}_{cned}$ * $P^{cned}$

- To find $T^{w}_{wned}$ is to project (or to calculate the `dot-product` between) each axis (as a unit vector) of $x^{wned}$, $y^{wned}$, $z^{wned}$, into the axis $x^w$, $y^w$, $z^w$.

- So we can get this matrix as:

```python
    T = np.array([
                  [0,1,0,0],
                  [0,0,1,0],
                  [1,0,0,0],
                  [0,0,0,1]], dtype=np.float32)
```


- And we have $T^{w}_{wned}$ = $T^{c}_{cned}$ = $T$.

## How to map the camera pose in NED to OpenCV Style: 

- OpenCV style camera-to-world pose is calculated as:

$T^{w}_{c}$ = $T^{w}_{wned}$ * $T^{wned}_{cned}$ * $T^{cned}_{c}$

- note: `$T^{w}_{c}$` etc are in LaTex style if not shown correctly.


## Apply Chain Rule
- We want to find the pose between `c` and `w` in OpenCV style coordinates;
- That is to say to find the cam-to-world pose $T^{w}_{c}$, which do the mapping $P^w = T^{w}_{c} * P^{c}$;
- Using the chain rule, we have:

$T^{w}_{c}$ = $T^{w}_{wned}$ * $T^{wned}_{cned}$ * $T^{cned}_{c}$ = $T$ * `camera-to-world-pose-NED` * inv(T)

where, we assume the `camera-to-wolrd pose in NED` is provided by the dataset (e.g., [TartanAir dataset](https://github.com/castacks/tartanair_tools/blob/b2f023bbca5606c05d4189811c3eee6f99564037/data_type.md)). Please see my [answer to this issue](https://github.com/castacks/tartanair_tools/issues/37) in the TartanAir Dataset repo.
