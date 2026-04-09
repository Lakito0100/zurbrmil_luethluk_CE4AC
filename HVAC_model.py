import numpy as np
import pandas as pd
import psychro as psy
import matplotlib.pyplot as plt

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
def ModelRecAir(m, α, β, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA):
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
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        TZ:     Thermal Zone
        BL:     Building
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..4    unknown points (temperature, humidity ratio)

        <----|<--------------------------------|
             |                                |
             |              |-------|         |
        -o->MX1--0->HC1--1->|       MX2--3->TZ--4-|
                    /       |       |       ||    |
                    |       |->AH-2-|       BL    |
                    |                             |
                    |<-----------Kθ---------------|<-t4


    Returns
    -------
    x       vector 13 elem.:
            θ0, w0, t1, w1, t2, w2, t3, w3, t4, w4,...
                QHC1, QsTZ, QlTZ

    """
    Kθ = 1e10                        # controller gain
    wO = psy.w(θO, φO)            # hum. out

    # Model
    θs0, Δ_θs = θS, 2             # initial guess saturation temp.

    A = np.zeros((13, 13))          # coefficents of unknowns
    b = np.zeros(13)                # vector of inputs
    while Δ_θs > 0.01:
        # MX1
        A[0, 0], A[0, 8], b[0] = m * c, -(1 - α) * m * c, α * m * c * θO
        A[1, 1], A[1, 9], b[1] = m * l, -(1 - α) * m * l, α * m * l * wO
        # HC1
        A[2, 0], A[2, 2], A[2, 10], b[2] = m * c, -m * c, 1, 0
        A[3, 1], A[3, 3], b[3] = m * l, -m * l, 0
        # AH
        A[4, 2], A[4, 3], A[4, 4], A[4, 5], b[4] = c, l, -c, -l, 0
        A[5, 4], A[5, 5] = psy.wsp(θs0), -1
        b[5] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[6, 2], A[6, 4], A[6, 6], b[6] = β * m * c, (1 - β) * m * c, -m * c, 0
        A[7, 3], A[7, 5], A[7, 7], b[7] = β * m * l, (1 - β) * m * l, -m * l, 0
        # TZ (connected directly from point 3)
        A[8, 6], A[8, 8], A[8, 11], b[8] = m * c, -m * c, 1, 0
        A[9, 7], A[9, 9], A[9, 12], b[9] = m * l, -m * l, 1, 0
        # BL
        A[10, 8], A[10, 11], b[10] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[11, 9], A[11, 12], b[11] = mi * l, 1, mi * l * wO + Qla
        # Kθ
        A[12, 8], A[12, 10], b[12] = Kθ, 1, Kθ * θIsp

        x = np.linalg.solve(A, b)
        Δ_θs = abs(θs0 - x[4])
        θs0 = x[4]
    return x


def RecAirCAV(α=1, β=0.1,
              θS=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
              Qsa=0, Qla=0, mi=2.18, UA=935.83):
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

        <----|<--------------------------------|
             |                                |
             |              |-------|         |
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
    None
    """
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
    print(f'm = {m: 5.3f} kg/s constant for design conditions:')
    print(f'    [θSd = {θS: 3.1f} °C, mi = 2.18 kg/S, θO = -1°C, φ0 = 100%]')

    # Model
    x = ModelRecAir(m, α, β,
                    θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA)

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

    psy.chartA(θ, w, A)

    θ = pd.Series(θ)
    w = 1000 * pd.Series(w)
    P = pd.concat([θ, w], axis=1)       # points
    P.columns = ['θ [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    Q = pd.Series(x[10:], index=['QsHC1', 'QsTZ', 'QlTZ'])
    # Q.columns = ['kW']
    pd.options.display.float_format = '{:,.2f}'.format
    print()
    print(Q.to_frame().T / 1000, 'kW')

    return None