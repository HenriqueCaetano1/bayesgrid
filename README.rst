========
bayesgrid
========

.. A python tool for generating synthetic, unbalanced power distribution systems using bayesian models

This tool provides a comprehensive, data-driven module to generate synthetic power distribution systems using Bayesian models. The models learn statistical properties from real grid data—such as component parameters, topological distances, and reliability metrics—to generate new, statistically similar, and realistic synthetic networks.

The primary inputs for generation are high-level topological descriptors, such as the *hop distance* (topological distance) of each bus and line from the substation. These inputs can be derived from existing ``pandapower`` networks or generated from scratch using real-world road networks from ``OpenStreetMap``.

The result is a set of complete, synthetic network parameters, including:

* **Bus-Level Data:**
    * 3-Phase Power (kW)
    * Phase Allocation (A, B, C, AB, ABC, etc.)
    * Failure Frequency (CAIFI)
    * Failure Duration (CAIDI)
* **Line-Level Data:**
    * Resistance (R1)
    * Reactance (X1)

A step-by-step tutorial on how to use the package can be found in the ``notebook_tutorials`` directory.

License
-------

The code in this repository is licensed under the **MIT License** (MIT).
See `LICENSE.txt <LICENSE.txt>`_ for rights and obligations.