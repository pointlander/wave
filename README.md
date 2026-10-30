# wave

A two-dimensional n-body simulation of a group of galaxies. Each galaxy is a Gaussian wave packet with a spatial representation and a Fourier dual. Gravity proposes the next state. Monte Carlo tree search around that proposal keeps the single lowest-scoring state, and leaves the proposal only when the lower block entropy pays for the move.

## Units

The default run is a group of 12 Milky Way-scale halos in a periodic 3 Mpc cell. Newton's constant is

`G = 4.3009172706×10⁻⁶ kpc M_sun⁻¹ (km/s)²`.

| Quantity | Unit | Default |
| --- | --- | --- |
| length | kpc | box 3000, group radius 540 |
| mass | M_sun | 1×10¹² per galaxy |
| velocity | km/s | about 310 at the group edge |
| time | Gyr | step 0.139, 24 steps span 3.33 |
| packet width `σ` | kpc | 120 |
| Plummer softening | kpc | 60 |
| energy | M_sun (km/s)² | |

These values are the original stable search scaled onto that box. The velocity scale is the one that makes the tuned force law match Newton's `G` at 10¹² solar masses. The carrier scale `ħ` is about 1.27×10⁴ kpc km/s, so `K = V / ħ` has the same number of radians across the box as that tuned model. It is not the quantum constant.

## Model

Galaxies live in a periodic box. Gravity is Plummer-softened Newtonian gravity in the plane, integrated with velocity Verlet. Total momentum is held fixed. The integrator advances positions in kpc and velocities in km/s, so its internal time step is kpc/(km/s); the table converts that to Gyr.

Each galaxy is a Gaussian envelope of width `σ` centered at `(X, Y)`, moving with velocity `V`. Its frequency representation is the matching Gaussian in wavevector space:

- carrier `K = V / ħ`
- spectral width `σ_k = 1 / (2σ)`
- an internal phase `φ`

The field `Ψ` is the coherent sum of the packets. `H_spatial` is the von Neumann entropy of the entanglement between `x` and `y`, divided by `log(grid)`. A product wave scores 0 and a maximally entangled wave scores 1.

`H_block` splits that grid into blocks of 8 cells on a side, or half the grid when that is smaller. It is the von Neumann entropy of the entanglement between the block index and the offset inside the block, divided by the log of the smaller factor. Blocks that carry the same interior wave score 0, and a wave that differs from block to block scores higher. A Fourier transform inside each block, taken separately along each axis, leaves `H_block` unchanged, so it is computed from `Ψ` directly.

Spatial entropy is measured and reported, and it does not enter the score. The search score is `H_block` plus `0.03` times the mean squared distance from the Newtonian proposal, measured in units of the mutation scales. Lower is better. A move is kept only when the drop in block entropy exceeds that charge, and the printed entropies do not include it. A non-finite block entropy is skipped.

The search is a Monte Carlo tree search rooted at the Newtonian proposal. A move mutates position, velocity, and phase, then restores total momentum. Progressive widening limits how many children a node grows. Selection uses UCB on the average of each node's simulation returns, and a simulation returns the lowest score it reached. Tree expansion adds one mutation, then one more ply. Each rollout step scores 4 mutations and continues from the lowest score. The two new plies and that batch are scored at the same time, one frame per core. That winner is added to the tree so a later iteration can expand from it. The other samples remain candidates and are not expanded. The step keeps the single best-scoring frame. The earliest frame wins a tie, and the proposal is scored first, so it is kept when no move pays for itself.

The initial condition is a rotating disk of equal-mass galaxies with random phases.

## Build and run

Requires Go 1.27. [ffmpeg](https://ffmpeg.org/) with `libvpx-vp9` is required only for `-webm`.

```
go test ./...
go run .
```

A default run prints one row per step: spatial entropy, block entropy, the Newtonian proposal's block entropy, mean separation in kpc, kinetic energy, and potential energy. Time in that table is Gyr. The entropies are measured on the accepted frame. `agree` is 1 when that frame matches the Newtonian proposal, and `agree_rate` is how often that has happened so far. The summary reports the same fraction for the whole run.

Run for 30 seconds of real time and write a VP9 WebM. Frames show `|Ψ|²` on the left and the fftshifted spectrum `|Ψ̂|²` on the right. `-duration` overrides `-steps`. Playback length is `frames / fps`, which is not the same as the wall-clock limit.

```
go run . -duration 30s -webm wave.webm
```

Write a PNG sequence and `trace.csv` as well:

```
go run . -duration 30s -webm wave.webm -out frames
```

`-duration` accepts a Go duration (`30s`, `1m`) or a bare number of seconds (`30`).

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-n` | 12 | number of galaxies |
| `-mass` | 1e12 | galaxy mass in solar masses |
| `-box` | 3000 | periodic box length in kpc |
| `-steps` | 24 | steps after the initial state, used when `-duration` is 0 |
| `-duration` | 0 | wall-clock limit; overrides `-steps` |
| `-dt` | 0.139 | Newtonian time step in Gyr |
| `-grid` | 64 | field size, a power of two |
| `-iters` | 48 | Monte Carlo iterations per step |
| `-widen` | 1 | progressive widening coefficient |
| `-explore` | 0.15 | UCB exploration constant |
| `-rollout` | 2 | rollout steps after each new node; each step keeps the best of 4 mutations |
| `-seed` | 1 | random seed |
| `-gravity` | 4.301e-6 | `G` in kpc M_sun⁻¹ (km/s)² |
| `-softening` | 60 | Plummer softening length in kpc |
| `-hbar` | 1.270e4 | carrier scale in kpc km/s, `K = V / ħ` |
| `-sigma` | 120 | packet width in kpc |
| `-radius` | 540 | initial group radius in kpc |
| `-mutpos` | 30 | position mutation scale in kpc |
| `-mutvel` | 8.47 | velocity mutation scale in km/s |
| `-mutphase` | 0.2 | phase mutation scale in radians |
| `-out` | | directory for PNG frames and `trace.csv` |
| `-webm` | | path of a VP9 WebM written by ffmpeg |
| `-fps` | 12 | WebM frame rate |

Set the mutation scales to 0 to take the Newtonian step unchanged.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
