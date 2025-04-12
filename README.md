# Inverse_Bloch
Create new RF pulse shapes by fitting solutions to Bloch Equations.

This repository contains a Python package (under development) that uses neural networks to fit custom RF pulse and field gradient shapes to a given slice profile. The code is based on a (adapted and optimised) solver for the Bloch Equations written by [Will Grissom](https://www.vanderbilt.edu/vise/visepeople/will-grissom/). The training workflow is as follows:

* Initialize a random pulse and gradient shape as `pulse, gradient = model(t)`, where model is a chosen neural network,
* pass `pulse`, `gradient` to the Bloch solver to obtain the corresponding frequency profile of $M_z$ and $M_{xy}$,
* compute the $L^2$ error between $M_z$, $M_{xy}$ and a prescribed target profile,
* backpropagate to update the parameters of `model`.

This procedure yields pulse/gradient pairs, which approximate the prescribed targets. Further constraints (e.g. on the slope of `gradient` or the phase of $M_{xy}$ can be prescribed by modifying the loss function accordingly.
