#!/bin/bash

./installation.sh
EXP_ID="BATCH_$(date +'%Y-%m-%d-%H-%M-%S')"
echo "Starting Experiment Batch: $EXP_ID"

alpha_values=("0.1")
datasets=("anaheim" "education")

sos_options=("")
iw_options=("")
methods=("graph_rlcp")
kmm_b=("1")
convex_dist_coeffs=("1.0")
ppr_ks=("30")
convex_coeff_methods=("degree")
graph_densification_method=("rbf")
conformal_eval_method=("fsc" "wsc" "marginal")
wm="convex"

bandwidth_values=("2.0")
graph_densification_threshold=("0.5")


# --- 3. Run Experiments ---
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for bandwidth in "${bandwidth_values[@]}"; do
            for alpha in "${alpha_values[@]}"; do
                for sos_flag in "${sos_options[@]}"; do
                    for iw in "${iw_options[@]}"; do
                        for k_b in "${kmm_b[@]}"; do
                            for ccm in "${convex_coeff_methods[@]}"; do
                                for gdt in "${graph_densification_threshold[@]}"; do
                                    for gdm in "${graph_densification_method[@]}"; do
                                        for ppr_k in "${ppr_ks[@]}"; do
                                            python main.py \
                                            -cff ./config/$dataset.yaml \
                                            -a $alpha \
                                            -m $method \
                                            -b $bandwidth \
                                            -dtd id \
                                            $sos_flag \
                                            -eid $EXP_ID \
                                            -wdb \
                                            $wss_flag \
                                            -tt 10 \
                                            -pk $ppr_k \
                                            -cdc 1.0 \
                                            -gdt $gdt \
                                            -ccm $ccm \
                                            -gdm $gdm \
                                            -cem ${conformal_eval_method[*]} \
                                            -dnp 3 \
                                            -wm $wm \
                                            -wnv 100
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done