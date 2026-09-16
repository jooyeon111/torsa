#include "Optimizer.hpp"
#include <iostream>
#include <string>

// Driver for the SRAM-combination (and cd) optimizer.
//   optimizer <arrayConfig.cfg> <layer.cfg> [sramDir] [objective]
//     arrayConfig : array_config/*.cfg (Array_cfg)
//     layer       : layer/*.cfg        (layer_cfg)
//     sramDir     : CACTI results root (default: cacti_results)
//     objective   : energy (default) | cycles | edp | tops; unrecognized values fall back to energy
//   env OPT_TOPK   : number of top analytic candidates re-evaluated with DRAMsim3 (default 8, 0 = analytic only)
// Exit status: 0 = success, 1 = usage error or exception, 2 = no feasible combination.
//
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <arrayConfig.cfg> <layer.cfg> [sramDir] [objective]\n"
            << "  sramDir   : CACTI results root (default: cacti_results)\n"
            << "  objective : energy (default) | cycles | edp | tops (maximize TOPS/W/mm^2)\n"
            << "  env OPT_TOPK : after analytic ranking, refine only the top K with DRAMsim3 (default 8, 0=analytic-only)\n";
        return 1;
    }

    try {
        const std::string sramDir = (argc >= 4) ? argv[3] : "cacti_results";
        Optimizer opt(argv[1], argv[2], sramDir);

        if (argc >= 5) {
            const std::string o = argv[4];
            if      (o == "cycles") opt.setObjective(Optimizer::Objective::Cycles);
            else if (o == "edp")    opt.setObjective(Optimizer::Objective::EnergyDelayProduct);
            else if (o == "tops")   opt.setObjective(Optimizer::Objective::TopsPerWattPerMm2);
            else                    opt.setObjective(Optimizer::Objective::TotalEnergy);
        }

        opt.optimize();
        opt.printReport(std::cout, 10);
        if (!opt.hasFeasible()) return 2;   // no feasible combination.
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
