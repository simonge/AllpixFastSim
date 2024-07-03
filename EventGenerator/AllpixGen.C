#include <iostream>
#include <fstream>
#include <HepMC3/GenEvent.h>
#include <HepMC3/GenVertex.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/WriterRootTree.h>
#include <TMath.h>
#include <TRandom3.h>

using namespace HepMC3;

int AllpixGen(int nEvents = 10000000) {
//int AllpixGen(int nEvents = 100) {

    // Time limit (ns)
    double time = 25.0;
    double x_size = 0.055;
    double y_size = 0.055;

    // Electron mass (MeV)
    double mass = 0.511;  // MeV

    // Electron momentum magnitude (GeV)
    double momentum = 1;  // GeV

    // Maximum Pt
    double max_Pt = 0.1;  // GeV


    // Open root output file for hepmc events
    WriterRootTree  root_output("/scratch/EIC/Events/Allpix2/Allpix2_Events.root");

    // Loop over the number of events
    for (size_t i = 0; i < nEvents; i++)
    {           
        // Create a new HepMC event
        GenEvent event(Units::GEV,Units::MM);

        // Set event number
        event.set_event_number(i);

        // Random time between 0 and the limit
        double t = gRandom->Uniform(0.0, time);

        // Random position on the surface of 1/8th of a pixel
        double x = x_size;
        double y = 0.0;
        double z = 0.0;

        while (x > y)
        {
            x = gRandom->Uniform(0, x_size/2);
            y = gRandom->Uniform(0, y_size/2);
        }
      
        FourVector pos(x, y, z, t);

        // Create a vertex
        GenVertexPtr vertex = std::make_shared<GenVertex>(pos);

        // Generate random x/ycompoents of the momentum until their combined magnitude is less than max_Pt
        double px = max_Pt;
        double py = max_Pt;
        while (TMath::Sqrt(px * px + py * py) > max_Pt)
        {
            px = gRandom->Uniform(-max_Pt, max_Pt);
            py = gRandom->Uniform(-max_Pt, max_Pt);
        }


        double pz  = -TMath::Sqrt(momentum * momentum - px * px - py * py);
        double E   = TMath::Sqrt(momentum * momentum + mass * mass);

        FourVector mom(px, py, pz, E);

        // Create an electron particle
        GenParticlePtr electron = std::make_shared<GenParticle>(
            mom,  // Momentum and energy of the electron
            11,  // PDG ID of the electron
            1  // Status code of the electron
        );
        vertex->add_particle_out(electron);

        event.add_vertex(vertex);

        root_output.write_event(event);

    }
    root_output.close();

    return 0;
}