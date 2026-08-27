#!/usr/bin/env python3
                       
   
                                                                  

                                                         

                                                                            
                                                    
                      
   
                                       
                                                                       
                                             

                                                                              
                                                          

            
                                     
                                                    
   
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

PI = math.pi
C2 = 1.438776877e-2       
SMALL = 1.0e-300


def trapezoid(values, coordinates):
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(values, coordinates))


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


def alumina_refractive_index(lam, T):
                                                                                      
    lam_um = lam*1.0e6
    l2 = lam_um*lam_um

    val = 1.0 + l2*(
        1.024/(l2 - 0.00376)
      + 1.058/(l2 - 0.01225)
      + 5.281/(l2 - 321.4)
    )

    n_base = math.sqrt(max(0.0, val))
    n = n_base*(0.9904 + 2.02e-5*T)

    k = (
        0.002
       *(0.06*l2 + 0.7*lam_um + 1.0)
       *math.exp(1.847*(T/1000.0 - 2.95))
    )

                                                                    
    return complex(n, -max(k, 0.0))


def planck_weight_lambda(lam, T):
                                                                                 
    if T <= 0.0 or lam <= 0.0:
        return 0.0

    y = C2/(lam*T)
    if y > 700.0:
        return 0.0

    denom = math.expm1(y)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0

    return 1.0/(lam**5 * denom)


def mie_coefficients(x, m):
       
                                      

                                                                              
                                                                         
                                                                            
       
    if x <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(2, complex), np.zeros(2, complex)

    mx_mag = abs(m*x)
    n_stop = max(1, int(x + 4.0*(x**(1.0/3.0)) + 2.0))
    n_max = max(n_stop, int(mx_mag)) + 15

    D = [0j]*(n_max + 2)
    z = m*x
    for n in range(n_max, 0, -1):
        rn_over_z = n/z
        D[n-1] = rn_over_z - 1.0/(D[n] + rn_over_z)

    an = np.zeros(n_stop + 2, dtype=np.complex128)
    bn = np.zeros(n_stop + 2, dtype=np.complex128)

    psi_nm1 = math.sin(x)
    psi_n = math.sin(x)/x - math.cos(x)

    xi_nm1 = complex(psi_nm1, math.cos(x))
    xi_n = complex(psi_n, math.cos(x)/x + math.sin(x))

    q_ext_sum = 0.0
    q_sca_sum = 0.0

    for n in range(1, n_stop + 1):
        rn = float(n)
        cn = float(2*n + 1)

        alpha = D[n]/m + rn/x
        beta = m*D[n] + rn/x

        an[n] = (alpha*psi_n - psi_nm1)/(alpha*xi_n - xi_nm1)
        bn[n] = (beta*psi_n - psi_nm1)/(beta*xi_n - xi_nm1)

        q_ext_sum += cn*(an[n] + bn[n]).real
        q_sca_sum += cn*(abs(an[n])**2 + abs(bn[n])**2)

        psi_np1 = ((2*n + 1)/x)*psi_n - psi_nm1
        xi_np1 = ((2*n + 1)/x)*xi_n - xi_nm1

        psi_nm1, psi_n = psi_n, psi_np1
        xi_nm1, xi_n = xi_n, xi_np1

    q_ext = max(0.0, 2.0*q_ext_sum/(x*x))
    q_sca = max(0.0, 2.0*q_sca_sum/(x*x))
    q_abs = max(0.0, q_ext - q_sca)

    g = 0.0
    if q_sca > 1.0e-300:
        g_sum = 0.0

        for n in range(1, n_stop):
            rn = float(n)
            g_sum += (
                rn*(rn + 2.0)/(rn + 1.0)
               *((an[n]*np.conj(an[n+1])).real + (bn[n]*np.conj(bn[n+1])).real)
            )

        for n in range(1, n_stop + 1):
            rn = float(n)
            g_sum += (
                (float(2*n + 1)/(rn*(rn + 1.0)))
               *(an[n]*np.conj(bn[n])).real
            )

        g = clamp(4.0*g_sum/(x*x*q_sca), -0.999, 0.999)

    return q_ext, q_sca, q_abs, g, an, bn


def mie_phase_raw(x, q_sca, an, bn, mu_values):
       
                                                         

                                        
                
                
                                          
                                                          

                                                                           
                          
       
    n_stop = len(an) - 2

    if n_stop < 1 or q_sca <= 1.0e-300 or x <= 0.0:
        return np.ones_like(mu_values)

    mu = np.clip(mu_values, -1.0, 1.0)

    pi_nm1 = np.zeros_like(mu)        
    pi_n = np.ones_like(mu)           

    S1 = np.zeros_like(mu, dtype=np.complex128)
    S2 = np.zeros_like(mu, dtype=np.complex128)

    for n in range(1, n_stop + 1):
        rn = float(n)
        tau_n = rn*mu*pi_n - float(n + 1)*pi_nm1

        coef = float(2*n + 1)/(rn*float(n + 1))
        S1 += coef*(an[n]*pi_n + bn[n]*tau_n)
        S2 += coef*(an[n]*tau_n + bn[n]*pi_n)

        pi_np1 = (float(2*n + 1)/rn)*mu*pi_n - (float(n + 1)/rn)*pi_nm1
        pi_nm1, pi_n = pi_n, pi_np1

                                                                                
                                                                                 
    phi = 2.0*(np.abs(S1)**2 + np.abs(S2)**2)/(x*x*max(q_sca, 1.0e-300))
    phi = np.maximum(phi.real, 0.0)
    return phi


def normalize_phi(phi, mu_values):
    integ = trapezoid(phi, mu_values)
    if not math.isfinite(integ) or integ <= 1.0e-300:
        return np.ones_like(phi)
    return phi*(2.0/integ)


def phase_moments(phi, mu_values):
    norm = trapezoid(phi, mu_values)
    g_phi = 0.5*trapezoid(mu_values*phi, mu_values)
    return norm, g_phi


def logspace(lo, hi, n):
    if n <= 1:
        return np.array([lo], dtype=float)
    return lo*(hi/lo)**(np.arange(n, dtype=float)/(n - 1))


def linspace(lo, hi, n):
    return np.linspace(lo, hi, n, dtype=float)


def compute_node(d, T, mu_values, lambdas):
    intW = 0.0
    int_abs = 0.0
    int_sca = 0.0
    int_ext = 0.0

    int_scaW = 0.0
    int_g = 0.0
    phi_acc = np.zeros_like(mu_values)

    if len(lambdas) > 1:
        dlog = math.log(lambdas[-1]/lambdas[0])/(len(lambdas) - 1)
    else:
        dlog = 1.0

    for li, lam in enumerate(lambdas):
        trap = 0.5 if (li == 0 or li == len(lambdas) - 1) else 1.0

                                                                
        W = trap*lam*dlog*planck_weight_lambda(lam, T)

        if W <= 0.0 or not math.isfinite(W):
            continue

        m = alumina_refractive_index(lam, T)
        x = PI*d/lam

        qext, qsca, qabs, g, an, bn = mie_coefficients(x, m)

        intW += W
        int_abs += W*qabs
        int_sca += W*qsca
        int_ext += W*qext

        if qsca > 1.0e-300:
            Ws = W*qsca
            int_scaW += Ws
            int_g += Ws*g
            phi_acc += Ws*mie_phase_raw(x, qsca, an, bn, mu_values)

    if intW <= 1.0e-300:
        raise RuntimeError(f"Planck weight is zero for d={d}, T={T}")

    Qabs = int_abs/intW
    Qsca = int_sca/intW
    Qext = int_ext/intW

    if int_scaW > 1.0e-300:
        phi = normalize_phi(phi_acc/int_scaW, mu_values)
        g = clamp(int_g/int_scaW, -0.999, 0.999)
    else:
        phi = np.ones_like(mu_values)
        g = 0.0

    norm, g_phi = phase_moments(phi, mu_values)
    return Qabs, Qsca, Qext, g, phi, norm, g_phi


def planck_band_fraction(T, lambda_min, lambda_max):
                                                                       
    c2 = 1.438776877e-2

    def tail(x):
        total = 0.0
        n = 1
        while True:
            en = math.exp(-n*x)
            term = en*(x**3/n + 3*x*x/(n*n) + 6*x/(n**3) + 6/(n**4))
            total += term
            if abs(term) < 1.0e-15*max(1.0, abs(total)):
                return total
            n += 1
            if n > 100000:
                raise RuntimeError("Planck band integration did not converge")

    x_low = c2/(lambda_max*T)
    x_high = c2/(lambda_min*T)
    return (tail(x_low) - tail(x_high))/(PI**4/15.0)


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('--out', default='alumina_mieTable.dat')

    ap.add_argument('--t-min', type=float, default=300.0)
    ap.add_argument('--t-max', type=float, default=5000.0)
    ap.add_argument('--n-t', type=int, default=80)

    ap.add_argument('--d-min', type=float, default=10e-6)
    ap.add_argument('--d-max', type=float, default=400e-6)
    ap.add_argument('--n-d', type=int, default=20)

    ap.add_argument('--n-mu', type=int, default=5001)

    ap.add_argument('--lambda-min', type=float, default=0.5e-6)
    ap.add_argument('--lambda-max', type=float, default=8.0e-6)
    ap.add_argument('--n-lambda', type=int, default=120)

    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()

    if args.n_t < 1 or args.n_d < 1 or args.n_mu < 3 or args.n_lambda < 1:
        raise ValueError("Require n_t>=1, n_d>=1, n_mu>=3, n_lambda>=1")

    Tvals = linspace(args.t_min, args.t_max, args.n_t)
    Dvals = logspace(args.d_min, args.d_max, args.n_d)
    Muvals = linspace(-1.0, 1.0, args.n_mu)
    Lambdas = logspace(args.lambda_min, args.lambda_max, args.n_lambda)

    n_nodes = args.n_d*args.n_t
    print(f"MIE_TABLE -> {args.out}")
    print(f"nD={args.n_d}, nT={args.n_t}, nMu={args.n_mu}, nLambda={args.n_lambda}, nodes={n_nodes}")
    print("This can be expensive for large nD/nT/nMu/nLambda.")

    if args.dry_run:
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {out}; pass --force explicitly")

    generation_args = vars(args).copy()
    generation_args.pop('force', None)
    generation_args.pop('dry_run', None)
    generator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tmp = out.with_name(out.name + f'.tmp.{os.getpid()}')

    payload_hash = hashlib.sha1()
    with tmp.open('w', encoding='ascii', newline='\n') as f:
        def emit(line):
            f.write(line + '\n')
            payload_hash.update((line + '\n').encode('ascii'))

        f.write('MIE_TABLE\n')
        f.write('formatVersion 1\n')
        f.write('units diameter_m temperature_K wavelength_m\n')
        f.write('interpolation logDiameter_linearTemperature_linearMu\n')
        f.write(f'generatorSha256 {generator_sha}\n')
        f.write('generationArgs ' + json.dumps(generation_args, sort_keys=True, separators=(',', ':')) + '\n')
        f.write(f'diameter {args.d_min:.17e} {args.d_max:.17e} {args.n_d}\n')
        f.write(f'temperature {args.t_min:.17e} {args.t_max:.17e} {args.n_t}\n')
        f.write(f'mu -1.00000000000000000e+00 1.00000000000000000e+00 {args.n_mu}\n')
        f.write(f'wavelength {args.lambda_min:.17e} {args.lambda_max:.17e} {args.n_lambda}\n')
        emit('Tvals ' + ' '.join(f'{x:.17e}' for x in Tvals))
        emit('Dvals ' + ' '.join(f'{x:.17e}' for x in Dvals))
        emit('Muvals ' + ' '.join(f'{x:.17e}' for x in Muvals))
        emit('PlanckBandFractions ' + ' '.join(f'{planck_band_fraction(float(T), args.lambda_min, args.lambda_max):.17e}' for T in Tvals))
        emit('DATA')

        for di, d in enumerate(Dvals):
            print(f'diameter {di+1}/{len(Dvals)} d={d:.6e}', flush=True)

            for ti, T in enumerate(Tvals):
                Qabs, Qsca, Qext, g, phi, norm, g_phi = compute_node(
                    float(d),
                    float(T),
                    Muvals,
                    Lambdas
                )

                emit(f'NODE {di} {ti} {Qabs:.17e} {Qsca:.17e} {Qext:.17e}')
                emit('PHI ' + ' '.join(f'{x:.17e}' for x in phi))

        emit('END_MIE_TABLE')
        f.write(f'payloadSha1 {payload_hash.hexdigest()}\n')

    os.replace(tmp, out)
    print(f'Wrote {out} ({out.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
