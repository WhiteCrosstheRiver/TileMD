# E2 — multi-replica batching (design only)

Do not build this until S-G2. The gate is: throughput at R copies of a 32k system is at least 0.8 times the throughput of one system with 32k·R atoms.

One engine launch holds R independent Lennard-Jones systems. They share the cutoff, the timestep, and the kernel. They do not share positions, velocities, or neighbor lists. The batch index is an extra dimension on every per-atom array: slot `(replica, atom)`.

A replica may differ in seed, composition, or a stored strain of the box. The pair loop does not read another replica. Rebuilds stay per replica, because one system can cross the skin while the others have not.

The expected win is occupancy at 32k, where one system does not fill the GPU. The cost is R neighbor lists. If that list traffic exceeds the occupancy gain, the gate fails and the path is dropped.
