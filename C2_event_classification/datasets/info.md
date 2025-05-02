## Info on the variables in the files

 - There are four files corresponding to events produced in electron-positron ($e^- e^+$) collisions at the $Z$-threshold, i.e. a centre-of-mass energy of $\sqrt{s} \approx 91.2 \text{GeV}/c^2$, where the $Z$ boson is then forced to decay into quark-antiquark pairs which then hadronise:
   - `Zbb.root` - the $Z$ decays to a pair of $b$-quarks, $Z^0 \to b\bar{b}$ 
   - `Zcc.root` - the $Z$ decays to a pair of $c$-quarks, $Z^0 \to c\bar{c}$ 
   - `Zss.root` - the $Z$ decays to a pair of $s$-quarks, $Z^0 \to s\bar{s}$ 
   - `Zud.root` - the $Z$ decays to a pair of either $d$-quarks or $u$-quarks, $Z^0 \to d\bar{d}$ or $Z^0 \to u\bar{u}$.

 - The signatures for these different processes are subtely different. Typically $b$-quark states have longer lifetimes so will produce displaced vertices. This is true to some extent of $c$-quark states as well. The heavy states produced by $b$- and $c$-quarks are not stable so will decay to lighter states and produce missing energy and charged leptons.

 - Your task is to try and train a network which can distinguish the different processes.

 - There are variables relating to the "Thrust" vector. This is a reconstructed quantity based on finding the direction of the two quarks from the $Z$ boson. The $Z$ boson is produced at threshold, so is at rest, therefore the two quarks are back-to-back (pointing in opposite directions). The thrust vector points along the direction of one quark. There is one of these vectors (thus a single X, Y, Z coordinate) per event.

 - There are variables relating to the reconstrcuted particles in the event. These start with `Particle_`. There are different number of particles in each event (labelled with `nParticle`) and various properties of these are saved (including their ID, charge, 4-momentum etc.).

 - There are variables relating to the reconstrcuted vertcies in the event. These are locations at which two or more charged tracks approximately intersect. In this case the optimal position of their intersection is fitted to produce a vertex. These start with `Vertex_`. There are different number of vertices in each event (labelled with `nVertex`) and various properties of these are saved (including their position, chi2 fit quality and whether they are the primary vertex or not).

 - The full list of variables, with some description, is given in the table below. Note that some of this is duplicate information (for example the four-vector can be built from m, px, py, pz or e, pt, eta, phi).

| Name | Type | Description |
| ----------------------- | --------------- | ---------------------------------- |
| `Thrust_x,y,z`          | double          | The (x,y,z) components of the unit vector which points in the direction of the two quarks produced from the Z |
| `Thrust_xerr,yerr,zerr` | double          | The error on the above quantities |
| `nParticle`             | int             | The number of reconstructed particles in the event |
| `Particle_ID`           | array of int    | The Particle Data Group ID of each particle (see PDG MC numbering scheme) |
| `Particle_m`            | array of double | Mass of each particle |
| `Particle_e`            | array of double | Energy of each particle |
| `Particle_q`            | array of int    | Charge of each particle |
| `Particle_p`            | array of double | Scalar momentum of each particle |
| `Particle_pt`           | array of double | Transverse momentum of each particle |
| `Particle_px,py,pz`     | array of double | Momentum components (x,y,z) of each particle |
| `Particle_eta`          | array of double | Pseudorapidity of each particle |
| `Particle_phi`          | array of double | Azimuthal angle of each particle |
| `Particle_orivtx_ind`   | array of int    | Gives the index in the vertex vector from which this particle originates 
| `nVertex`               | int             | The number of reconstructed vertices in the event |
| `Vertex_ntracks`        | array of int    | The number of charged tracks originating from this vertex |
| `Vertex_chi2`           | array of double | The chi2 of the vertex fit (i.e. the vertex fit quality) |
| `Vertex_isPV`           | array of int    | A flag whether the vertex is the Primary Vertex (the collision point where the Z is produced and decays) or not (i.e. a subsequent decay vertex) |
| `Vertex_m`              | array of double | The invariant mass of the vertex (i.e. take the 4-vectors of each particle originate from this vertex, sum them and find the invariant mass |
| `Vertex_x,y,z`          | array of double | The position of the vertex |
| `Vertex_xerr,yerr,zerr` | array of double | The error on the vertex position |
| ----------------------- | --------------- | ---------------------------------- |

