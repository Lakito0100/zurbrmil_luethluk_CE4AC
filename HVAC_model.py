import numpy as np
import pandas as pd
import CE4AC_packages.psychro as psy
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# global variables
# UA = 935.83                 # bldg conductance
# θIsp, wIsp = 18, 6.22e-3    # indoor conditions

θOd = -1                    # outdoor design conditions
mid = 2.18                  # infiltration design

# constants
c = 1e3                     # air specific heat J/kg K
l = 2496e3                  # latent heat J/kg


# *****************************************
# RECYCLED AIR
# *****************************************
def ModelRecAir(m, α, β, eta, θS, θO, φO, Qsa, Qla, mi, UA, T0, Tc, rho, Vd):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations

    INPUTS:
        m       mass flow of supply dry air, kg/s
        α       mixing ratio of outdoor air, -
        β       by-pass factor of the adiabatic humidifier, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K

    System:
        MX1:    Mixing box
        HC:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        TZ:     Thermal Zone
        BL:     Building
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..4    unknown points (temperature, humidity ratio)

        <----|<-----------------------------------|
            |                                    |
            |              |-------|             |
        -o->MX1--0->HC--1-->|       MX2--3->TZ--4-|
                    /       |       |       ||    |
                    |       |->AH-2-|       BL    |
                    |                             |
                    |<-----------Kθ---------------|<-t4


    Returns
    -------
    x       vector 13 elem.:
            θ0, w0, t1, w1, t2, w2, t3, w3, t4, w4,...
                QsTZ, QlTZ

    """
    Q_HP, cycle = heat_pump_Qc(T0, Tc, eta, rho, Vd)   #Heat pump
    wO = psy.w(θO, φO)            # hum. out

    # Model
    θs0, Δ_θs = θS, 2             # initial guess saturation temp.

    A = np.zeros((12, 12))          # coefficents of unknowns
    b = np.zeros(12)                # vector of inputs
    while Δ_θs > 0.01:
        # MX1
        A[0, 0], A[0, 8], b[0] = m * c, -(1 - α) * m * c, α * m * c * θO
        A[1, 1], A[1, 9], b[1] = m * l, -(1 - α) * m * l, α * m * l * wO
        # HC
        A[2, 0], A[2, 2],  b[2] = -m * c, m * c, Q_HP
        A[3, 1], A[3, 3], b[3] = -m * l, m * l, 0
        # AH
        A[4, 2], A[4, 3], A[4, 4], A[4, 5], b[4] = c, l, -c, -l, 0
        A[5, 4], A[5, 5] = psy.wsp(θs0), -1
        b[5] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[6, 2], A[6, 4], A[6, 6], b[6] = β * m * c, (1 - β) * m * c, -m * c, 0
        A[7, 3], A[7, 5], A[7, 7], b[7] = β * m * l, (1 - β) * m * l, -m * l, 0
        # TZ (connected directly from point 3)
        A[8, 6], A[8, 8], A[8, 10], b[8] = m * c, -m * c, 1, 0
        A[9, 7], A[9, 9], A[9, 11], b[9] = m * l, -m * l, 1, 0
        # BL
        A[10, 8], A[10, 10], b[10] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[11, 9], A[11, 11], b[11] = mi * l, 1, mi * l * wO + Qla

        x = np.linalg.solve(A, b)
        Δ_θs = abs(θs0 - x[4])
        θs0 = x[4]
    return x, Q_HP, cycle


def RecAirCAV(α=1, β=0.1, eta=0.65,
            θS=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
            Qsa=0, Qla=0, mi=2.18, UA=935.83, 
            show_plots=True, show_output=True):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations

    INPUTS:
        α   mixing ratio of outdoor air, -
        β    by-pass factor of the adiabatic humidifier, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp  indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO    outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K

    System:
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        TZ:     Thermal Zone
        BL:     Building
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..4    5 unknown points (temperature, humidity ratio)

        <----|<-----------------------------------|
            |                                    |
            |              |-------|             |
        -o->MX1--0->HC1--1->|       MX2--3->TZ--4-|
                    |       |       |       ||    |
                    |       |->AH-2-|       BL    |
                    |                             |
                    |<-----------Kθ---------------|<-t4

    13 Unknowns
        0..4: 2*5 points (temperature, humidity ratio)
        QsHC1, QsTZ, QlTZ
    Returns
    -------
    θ, w, Q
    """
    if show_plots:
        plt.close('all')
    wO = psy.w(θO, φO)            # hum. out

    # Mass flow rate for design conditions
    # Supplay air mass flow rate
    # QsZ = UA*(θO - θIsp) + mi*c*(θO - θIsp)
    # m = - QsZ/(c*(θS - θIsp)
    # where
    # θO, wO = -1, 3.5e-3           # outdoor
    # θS = 30                       # supply air
    # mid = 2.18                     # infiltration
    QsZ = UA * (θOd - θIsp) + mid * c * (θOd - θIsp)
    m = - QsZ / (c * (θS - θIsp))
    if show_output:
        print(f'm = {m: 5.3f} kg/s constant for design conditions:')
        print(f'    [θSd = {θS: 3.1f} °C, mi = {mi: 5.3f} kg/s, θO = {θO: 3.1f}°C, φ0 = {φO: 3.1f}]')

    # Model
    
    x, Q_HP, cycle = ModelRecAir(
        m, α, β, eta,
        θS,
        θO, φO,
        Qsa, Qla, mi, UA, 
        T0=15,          
        Tc=30,      
        rho=1.7,   # density of CO2 at compressor entry
        Vd=0.003   # volume of compressor (m3)
    )

    θ = np.append(θO, x[0:10:2])
    w = np.append(wO, x[1:10:2])

    # Adjancy matrix
    # Points calc.  o   0   1   2   3   4       Elements
    # Points pplot  0   1   2   3   4   5       Elements
    A = np.array([[-1, +1, +0, +0, +0, -1],     # MX1
                [+0, -1, +1, +0, +0, +0],     # HC1
                [+0, +0, -1, +1, +0, +0],     # AH
                [+0, +0, -1, -1, +1, +0],     # MX2
                [+0, +0, +0, +0, -1, +1]])    # TZ

    if show_plots:
        psy.chartA(θ, w, A,t_range=np.arange(min(θ)-5, max(θ)+5, 0.1), w_range=np.arange(max(min(w)-0.005, 0), max(w)+0.005, 0.0001))

    θ = pd.Series(θ)
    w = 1000 * pd.Series(w)
    P = pd.concat([θ, w], axis=1)       # points
    P.columns = ['θ [°C]', 'w [g/kg]']


    Q = pd.Series(
        [Q_HP, x[10], x[11]],
        index=['Q_HP', 'QsTZ', 'QlTZ']
    )

    pd.options.display.float_format = '{:,.2f}'.format

    if show_output:
        output = P.to_string(formatters={
            't [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)
        print()
        print(Q.to_frame().T / 1000, 'kW')

    return θ, w, Q, cycle

def heat_pump_Qc(T0, Tc, eta, rho, Vd, refrigerant="CO2"):
    """
    Model:
        Heat Pump controlled by the efficiency of the compressor

    -----------------------
    Inputs:
        T0   : source temperature [°C]
        Tc   : sink temperature [°C]
        eta  : isentropic efficiency compressor [-]
        rho  : density refrigerant at compressor entry [kg/m3]
        Vd   : volume flow at compressor [m3/s]

    System:
        EVA	:	evaporator
        COM	:   compressor
        CON	:	condenser
        EXP	:	expansion valve 

                        ^
                        |Tc,out
            <-----------CON<----------|
            |                         |
        EXP                       COM
            |              mRef-->    |
            |---------->EVA---------->|
                        ^
                        | T0,in
    Unknowns
        Qc	:	heat load of sink
        Q0	:	heat load of source
        mRef:	refrigerant mass flow
    """

    # --- Kelvin ---
    T0K = T0 + 273.15
    TcK = Tc + 273.15

    # --- Zustand 1: Verdichtereintritt ---
    P1 = PropsSI("P", "T", T0K, "Q", 1, refrigerant)
    H1 = PropsSI("H", "T", T0K, "Q", 1, refrigerant)

    # --- Hochdruck (Gas cooler / Kondensator) ---
    P3 = PropsSI("P", "T", TcK, "Q", 0, refrigerant)
    P2 = P3

    # --- Isentrope Verdichtung ---
    S2s = PropsSI("S", "P|gas", P1, "T", T0K, refrigerant)
    T2s = PropsSI("T", "P", P3, "S", S2s, refrigerant)
    H2s = PropsSI("H", "P|gas", P3, "T", T2s, refrigerant)

    # --- Reale Verdichtung ---
    H2 = H1 + (H2s - H1) / eta

    # --- Zustand 3: Gaskühleraustritt ---
    H3 = PropsSI("H", "P", P3, "Q", 0, refrigerant)
    
    # --- Massenstrom ---
    mRef = rho * eta * Vd * (3000/60)

    # --- Heizleistung ---
    Qc = mRef * (H2 - H3)

    return Qc, [[P1, H1], [P2, H2], [P3, H3], [P1, H3]]


