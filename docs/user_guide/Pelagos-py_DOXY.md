# Oxygen Processing: Aanderaa Optodes

```{image} ../_static/doxy/doxy_section.png
:alt: Oxygen concentration against pressure and time for the demo deployment
:width: 100%
```

## Overview

Dissolved oxygen in seawater gives us information about ocean ventilation, primary productivity, and remineralisation of organic matter. It is therefore considered a key variable in understanding the marine carbon cycle, the biological carbon pump, and evolution of water masses. Oxygen sensors have been integrated on a range of autonomous platforms, such as floats, gliders, and ALRs. pelagos-py implements an oxygen processing workflow, including calculations that convert raw sensor signals into oxygen concentrations, corrections, quality control, and standardised unit conversions. The workflow presented here is for Aanderaa oxygen optodes only, and has been designed according to best practice protocols (Bittig et al., 2016) and BGC-Argo protocols (Thierry et al., 2025a) that have been adapted where necessary to support a wide range of autonomous observing platforms and their unique capabilities. Default processing parameters are provided as starting points, but should be critically assessed by the user. This documentation is intended to guide these decisions.

*How does an optode measure dissolved oxygen?*

Dissolved oxygen is measured by optodes based on the principle of luminescence quenching by oxygen molecules. The optode has a sensing foil that is permeable to oxygen, allowing it to diffuse across it. The foil contains luminophores that are excited by short pulses of blue light. When the luminophores relax to their ground state, they release energy in the form of light, i.e., luminescence. When oxygen molecules collide with the luminophores, a part of the released energy is transferred to the oxygen molecule, thereby quenching the luminescence. The amount by which the luminescence is quenched, i.e. the luminescence intensity is reduced, can therefore be translated into dissolved oxygen concentration.

<!-- Hidden until the image and permission to reuse it are available.
```{figure} ../_static/doxy/optode_diagram_Aanderaa2007.png
:alt: Schematic illustration of the oxygen optode
:width: 70%

Schematic illustration of the oxygen optode. From Aanderaa (2007).
```
-->

However, since light intensity is affected by source intensity, scattering, and other variables, it is prone to biases. Instead, a single-frequency phase shift technique is used which relies on the exponential decay of luminescence with time. Excitation intensity is modulated at a constant frequency, producing an emission that can be described as a *phase*. When oxygen molecules quench the luminenscence pulses, the resulting phase of the emitted light is *shifted*. It is this *phase shift* that is used as the measurement of dissolved oxygen. More detailed information can be found in Bittig et al. (2018).

```{figure} ../_static/doxy/diagram_phase_shift_Bittig2018.png
:alt: Schematic illustration of the optode phase shift
:width: 90%

Schematic illustration of the optode phase shift. Reproduced from Bittig et al. (2018), Fig. 2, under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.
```

The integration of oxygen sensors onto autonomous platforms has greatly increased the spatial and temporal range on which oxygen variability can be studied and understood. However, the data output of the sensor requires rigorous QC, various corrections, calibrations, and processing by the user. The processing steps may vary depending on the optode type and available calibration data. pelagos-py suggests which processing steps need to be applied given the metadata in the data attributes. The toolbox will also suggest default parameters where it has enough information to do so, which can altered by the user. Differing nomenclature for the same variables outputted by different sensors have been standardised in the OG format, allowing pelagos-py to streamline processing steps as much as possible. Table 1 in the section on *Optode Variables* summarises how the standardisation of the optode variable names and how they are referred to in pelagos-py.

The workflow for processing dissolved oxygen measurements consists of the following steps:

1. **Temperature check** - applies quality checks on the optode temperature data.

2. **Uncalibrated phase shift** - calculates the uncalibrated phase shift by subtracting the red phase shift from the blue phase shift. If the red phase shift is not included in the dataset or is equal to zero, then the uncalibrated phase shift is simply the blue phase shift.

3. **Geometric lag correction** - applies a geometric lag correction that aligns the optode measurements to the CTD measurements as the autonomous platform descends/ascends in the water column. The CTD and optode may be mounted at different locations on different parts of the platform, which means the CTD and optode encounter the same parcel of water at different times. This time delay can lead to offsets in processed oxygen data, especially in strong temperature and salinity gradients. CTD and optode measurements are aligned to correct for this offset.

4. **Pressure correction on phase shift** - applies a pressure correction on the uncalibrated phase shift to correct for the effect of pressure on the luminescence lifetime of the luminophores in the foil.

5. **Calibrated phase shift** - converts the uncalibrated phase shift into the calibrated phase shift using calibration coefficients provided by the manufacturer. This convertion includes a temperature compensation.

6. **Molar oxygen concentration** - converts the calibrated phase shift into molar oxygen concentration using one of several given methods. Choice of method depends on sensor type and availability of manufacturer coefficients. pelagos-py will choose one of the methods based on the available metadata in the data attributes. The resulting molar oxygen concentration is representative of dissolved oxygen concentration in freshwater, i.e., for salinity = 0.

7. **Salinity correction** - applies a salinity correction to the molar oxygen concentration, resulting in a molar oxygen concentration that is representative of the salinity of the measured water parcel.

8. **Pressure correction on molar oxygen concentration** - applies a pressure correction on the molar oxygen concentration to account for the effect of pressure on quenching and on membrane O$_2$ equilibrium level.

9. **Dissolved oxygen content** - converts molar oxygen concentration ($\mu$mol L$^{-1}$) or oxygen partial pressure ($\mu$atm) to oxygen content ($\mu$mol kg$^{-1}$)

10. **Oxygen QC** - applies quality checks on oxygen content.

11. **Response correction** - applies a correction to account for the optode response time. *not yet implemented*

12. **In-air calibration** - *not yet implemented*

13. **Two-point calibration** - *not yet implemented*

```{image} ../_static/doxy/Pelagos-py_DOXY_Flowchart.png
:alt: Oxygen optode processing flowchart
:width: 100%
```


---


# Input Data

Processing dissolved oxygen measurements requires measurements from both the oxygen optode, the CTD, and several optional glider variables.

## Optode variables

- `TIME` — measurement time
- `BPHASE_DOXY` - phase shift measurement due to blue light excitation [$^{\circ}$]
- `RPHASE_DOXY` - phase shift measurement due to red light excitation (might not exist in dataset, in which case `RPHASE_DOXY` = 0) [$^{\circ}$]
- `CAL_PHASE_DOXY` — calibrated optode phase measurement [$\mu$s] (not required if `BPHASE_DOXY` is available)
- `MOLAR_DOXY` - dissolved oxygen concentration [$\mu$mol L$^{-1}$] | uncorrected for pressure and salinity if outputted directly by the sensor
- `TEMP_DOXY` — optode temperature [$^{\circ}$C]
- Optode calibration coefficients from the manufacturer calibration sheet

Different optodes give output of the same variables using different nomenclature. These have been standardised in OG format. The table below:

| Optode | Blue phase shift| Red phase shift | Uncalibrated phase shift | Calibrated phase shift | Molar oxygen concentration |
| -------------| :-------------: |:------------:| :-------------:|:-------------:|:-------------:|
| Aanderaa 3930, 3835, 3975, 4130, 4175, 3830      | BPHASE | RPHASE |              | DPHASE | MOLAR_DOXY |
| Aanderaa 4330, 4831, 48351     | C1PHASE |  C2PHASE| TPHASE | CALPHASE | MOLAR_DOXY
| **OG / pelagos-py** | **BPHASE_DOXY** | **PHASE_DOXY** | **UNCAL_PHASE_DOXY** | **CAL_PHASE_DOXY** | **MOLAR_DOXY** |

NRT datasets might contain different optode variables compared to recovered datasets. Depending on which optode variables are available, the starting point in the optode pipeline will differ. Note that starting lower down the pipeline sometimes means not all corrections can be applied.
> If `BPHASE_DOXY` is available, start at step 1 of the workflow.
>> If only `UNCAL_PHASE_DOXY` is available, apply step 1 and continue with step 3 of the workflow.
>>> If only `CAL_PHASE_DOXY` is available, apply steps 1 and 3 and continue with step 5 of the workflow.
>>>> If only `MOLAR_DOXY` is available, apply steps 1 and 3 and continue with step 6 of the workflow.


## CTD variables

- `TIME` — measurement time
- `TEMP` — CTD temperature [$^{\circ}$C]
- `PRAC_SALINITY` — practical salinity [psu]
- `PRES` — pressure [dbar]

## Glider variables

The geometric lag correction can be done using a default time value, or it can be estimated for each cast using the following variables:

- `vert_gradient` - vertical velocity of the platform, e.g. of the glider (optional) | determined by pelagos-py
- `GLIDER_PITCH` - pitch of the platform, e.g. of the glider (optional)
- `PROFILE_DIRECTION` or `SCI_PHASE` - variable distinguising up- and downcasts

## Calibration coefficients

Conversions of raw data into meaningful oxygen concentration values requires calibration values that are provided by the manufacturer. Calibration coefficients are either stored in the data attributes or can be given by the user.

---

# 1. Temperature QC

The optode has its own temperature measurements (`TEMP_DOXY`). For Aanderaa 3830 optodes and similar, these temperature values should generally not be used as the response time of the temperature sensor was too slow for these optode series. For Aanderaa 4330 optodes and similar, `TEMP_DOXY` will be used for part of the processing pipeline, and some prior inital QC should be done.

## 1.1 Global range check

Optode temperature should be within the following value range, and flagged as "bad data" if not:

$$-2.5 \leq TEMP\_DOXY \leq 40.0 \quad [^\circ\mathrm{C}]$$

Default configuration:

```yaml
parameters:
      qc_settings:
        range qc:
          variable_ranges:
            DOXY_TEMP:
              4: [-2.5, 40, outside]
```

## 1.2 Regional range check

Values outside physically plausible regional ranges for the Red Sea and the Mediterranean Sea should be flagged as "bad data" (i.e., QC flag 4). Note that oxygen values need to be expressed in oxygen content in $\mu$mol kg$^{-1}$ for this step.

<font color=orange> Implemented? </font>

Red Sea:
$$21.7 \leq TEMP\_DOXY \leq 40.0 \quad [^\circ\mathrm{C}]$$

Mediterranean Sea:
$$10.0 \leq TEMP\_DOXY \leq 40.0 \quad [^\circ\mathrm{C}]$$


## 1.3 Spike check

A spike check is applied using a test value:

Test value = | V2 - (V3 + V1)/2 | - | (V3 - V1) / 2 |

Where V2 is the measurement being tested as a spike, V1 and V3 are the values above and below.

**TEMP_OXY:** bad data when:
- Test value > 6 at pressure < 500 dbar
- Test value > 2 at pressure ≥ 500 dbar

## 1.4 Gradient test

A gradient check is applied using a test value:

Test value = | V2 − (V3 + V1)/2 |

where V2 is the measurement being tested as a spike, and V1 and V3 are the values above and below.

**For TEMP_DOXY:** the V2 value is flagged when:
- Test value exceeds 9 °C for pressures less than 500 dbar
- Test value exceeds 3 °C for pressures ≥ 500 dbar


---

# 2. Uncalibrated phase shift

The uncalibrated phase measurement (`UNCAL_PHASE_DOXY`) is defined and calculated as the difference between the phase from the blue light excitation (`BPHASE_DOXY`) and the red light excitation (`RPHASE_DOXY`).

$$
UNCAL\_PHASE\_DOXY = BPHASE\_DOXY - RPHASE\_DOXY
$$

The red light excitation is often not used, in which it does not appear in the dataset or is set to zero, and `UNCAL_PHASE_DOXY` = `BPHASE_DOXY`.



---

# 3. Geometric lag correction

The CTD and oxygen optode may not be physically co-located on the autonomous platform. For example, it is common for the CTD to be integrated nearer the middle or front of a glider, while the optode sits in the rear. The CTD therefore measures a water parcel at a different time than the optode measures the same parcel. Typically, this will lead to a time delay of several seconds. We can correct for this time offset by applying a geometric lag correction, which will align the optode data to the CTD data in time.

<!-- Hidden until the image and permission to reuse it are available.
```{figure} ../_static/doxy/diagram_geometric_alignment_WooGourcuff2023.png
:alt: Schematic illustration of the geometric lag correction
:width: 70%

Schematic illustration of the geometric lag correction. From Woo & Gourcuff (2023).
```
-->

If the distance between the sensors is $d$ and the glider's along-track velocity is $v$, the geometric time delay is

$$
\tau_{\mathrm{geom}} = \frac{d}{v}
$$

For a glider, the vertical component of velocity can be approximated by

$$
v = r\sin(\theta)
$$

where:

- $r$ = glider dive rate
- $\theta$ = glider pitch angle

Thus,

$$ \tau_{\mathrm{geom}} = \frac{d}{r\sin(\theta)} $$

For a known constant geometric delay, the optode data are interpolated to the time at which the corresponding water parcel reaches the CTD.

If the optode measurement occurs $\tau_{\mathrm{geom}}$ seconds after the CTD measurement, the optode time corresponding to a CTD observation at time $t$ is

$$ t_{\mathrm{optode}} = t_{\mathrm{CTD}} + \tau_{\mathrm{geom}} $$

The geometric lag correction can be applied using a default value for the lag (3.945), which is based on a distance between CTD and optode of 90 cm and the ideal conditions for glider travel ($r$ = 0.1 m s$^{-1}$, $\theta$ = 26$^{\circ}$).

Default configuration:

```yaml
lag_seconds: 3.945
distance_cm: 90
```

However, the pitch and dive rate can be far from ideal, resulting in values far from average. The user can therefore choose to let the toolbox calculate the mean geometric lag correction for each cast (Woo et al., 2023). This option will still use a default value for the distance between the sensors (90 cm), but this can be changed in the configuration file. The mean pitch and dive rate for each cast are used to calculate $\tau_{\mathrm{geom}}$, which is then passed through a low-pass filter to remove extreme values. The filtered values are then used as a linear time shift to the phase shift data.

```
Figure placeholder.
:alt: Example figure of mean geometric lag for each cast (separate for down and up cast), along with a line presenting the low-pass filter. Geometric lag on the y axis, time (or cast) on the y axis.
:width: 70%

**Placeholder.**
```


---

# 4. Pressure correction on phase shift

Pressure affects the luminescence lifetime of the luminophores, quenching, and membrane O2 equilibrium level (Bittig et al., 2015a; Bittig et al., 2016). To account for the pressure effect on luminescence lifetime, a hydrostatic pressure correction is applied to the uncalibrated phase shift (following Thierry et al., 2015a; also see Bittig et al., 2018):

$$UNCAL\_PHASE\_DOXY\_PCORR = UNCAL\_PHASE\_DOXY + Pcoef1 \frac{1}{1000} * PRES$$

`UNCAL_PHASE_DOXY_PCORR` is then the hydrostatic pressure corrected uncalibrated phase shift.

$PRES$ = Pressure in dbar.

$Pcoef1$ is a phase offset that depends on the sensing foil and on the manufacturer (see table below; Bittig et al., 2018). <font color=orange> These are provided in pelagos-py and will be automatically applied based on the sensor type in the attributes. The phase offset can also be changed by the user in the configuration file. </font>

| Sensor | $Pcoef1$ |
| -------------| :-------------:
| Aanderaa PSt3 foils     | 0.1$^{\circ}$ per 1000 dbar|
| SeaBird optodes | 0.115 $\mu$s per 1000 dbar |


Default configuration:

```yaml
correction_coefficient: 0.1
```

Note that the optode data should have been geometrically adjusted prior to this, and any other step that uses CTD data for processing optode data.

The pressure effect on quenching and on membrane O$_2$ equilibrium level will be corrected on the oxygen concentration: `MOLAR_DOXY`, which is done after the salinity correction.

---

# 5. Calibrated phase shift

The raw phase shift (`BPHASE_DOXY`) measured by the optode itself is dependent on temperature. For example, higher temperatures reduce the lifetime of the excited state of the luminophores. High temperatures also increase oxygen diffusivity across the foil, increasing the efficiency of luminescence quenching. The conversion of phase shift to oxygen concentration therefore needs to consider these temperature and oxygen dependencies, which are parameterised using results of elaborate calibration set-ups by the manufacturer that systematically vary oxygen and temperature across a range of values. The calibration coefficients inform mathematical models that are applied to the data. These are described in more detail in the next step that converts the phase shift into oxygen concentration.

However, each sensor responds differently to varying oxygen and temperature conditions due to differences in the luminophore foil, its thickness, batch, the LED, and electronics. This means that each sensor ideally requires its own set of calibration coefficients. In most new models, this is provided by the manufacturer. However, in some cases, sensors do not have their own individual multi-point calibration. Instead, only a few sensors of the same batch are calibrated for their O$_2$-T-response, the results of which are used as representative of the entire batch. To test how much each individual sensor within the same batch deviates from that batch-level multi-point calibration, manufacturers do a factory test on each sensor, the result of which is four coefficients, named phase coefficients represented on the calibrations sheet as *PhaseCoefs*, that describe how the individual sensors relate to the batch-level calibration in a 4th order polynomial model. This correction needs to be applied in the phase-domain, i.e., before the phase shift is converted into concentration. The result is the calibrated phase (`CAL_PHASE_DOXY`):

$$ CAL\_PHASE\_DOXY = c_0 + c_1* UNCAL\_PHASE\_DOXY + c_2* UNCAL\_PHASE\_DOXY^2 + c_3* UNCAL\_PHASE\_DOXY^3$$

where $c_0,\ldots,c_3$ are the phase calibration coefficients provided by the manufacturer (shown on the certificate as *PhaseCoef$_{0...3}$*) and are specific to the optode model. Phase coefficients c$_2$ and c$_3$ are usually zero, reducing the relationship to a linear function of the uncalibrated phase shift. For newer optode models, this step is often not required, because each sensors has its own multi-point calibration. In these cases, c$_0$ = 0 and c$_1$ = 1.

Default configuration:

```yaml
phase_coeffs: [0, 1, 0, 0]
```

<font color=orange> I changed the naming of the coefficients to reflect the calibration sheets. Discuss if/how to implement this in the toolbox. </font>

> This step is sometimes referred to as the "temperature-compensation" step, which is a term that is carried over historically, but is misleading since this step in the methods does not address the temperature dependence of the foil (this is done in the next step).

---

# 6. Molar oxygen concentration

As described above, the phase shift measured by the optode itself is dependent on temperature and the impact of temperature also differs across O$_2$ concentrations. The conversion of phase shift to oxygen concentration therefore needs to consider these temperature and oxygen dependencies, which are parameterised using results of elaborate calibration set-ups that systematically vary oxygen and temperature across a range of values. Results are translated into mathematical models that can be applied to the measured data. Different models/methods exist for optode types and are therefore described separately below, based on Thierry et al. (2025a) and Bittig et al. (2018). The choice of method depends on the available calibration data provided by the manufacturer. pelagos-py will choose a method based on the available coefficients in the data attributes, but both the choice of method and calibration coefficients can be altered by the user in the config file.

## 6.1 Aanderaa 3830 optodes

Important for these optodes is that the temperature values used in the conversion of phase shift (`CAL_PHASE_DOXY`) to oxygen concentration (`MOLAR_DOXY`) need to come from the CTD's temperature sensor `TEMP` (to which the optode data has been aligned in a previous step), *not* from the optode's temperature sensor (`TEMP_DOXY_SHIFTED`). This is because the temperature sensor of these older optodes had slow response times. <font color=orange> Based on the sensor type given in the data attributes, pelagos-py will select the appropriate temperature variable, but this can always be changed by the user. </font>

Default configuration:

```yaml
parameters:
    temperature_name: "TEMP"
```

### Stern-Volmer / SVU

This method uses the calibration coefficients derived from the multi-point calibration done that characterises the O$_2$-T dependence. The method shown here is only used in early shipments of these sensors:

$$
MOLAR\_DOXY_{S=0} = (\frac{F_1}{\frac{CAL\_PHASE\_DOXY}{K_0}-F_2}-1) K_1
$$

, where K$_0$ and K$_1$ are temperature dependent coefficients calculated using a 3rd degree polynomial with temperature as argument and coefficients provided in the calibration sheet, named *K0Coefs* and *K1Coefs*, respectively. F$_1$ and F$_2$ are temperature independent coefficients given in the calibration certification. `MOLAR_DOXY` is in $\mu$mol L$^{-1}$ and represents the oxygen concentration at salinity = 0.

```yaml
parameters:
    method: "SVU38"
    temperature_name: "TEMP"
    k0_coefs: [0, 0, 0, 0]
    k1_coefs: [0, 0, 0, 0]
    f_coefs: [0, 0]
```
<font color=orange> I have changed method names, but this has not been updated in pelagos yet. Have also added additional coefficient names but haven't yet checked how these have been implemented. </font>

### Polynomial

The calibrated phase can also be converted into oxygen concentration using 20 coefficients (provided by the manufacturer) that are sensor specific, in a 4$^{th}$ degree polynomial function:

$$
MOLAR\_DOXY_{S=0} = C_0 + C_1 . CAL\_PHASE\_DOXY + C_2 . CAL\_PHASE\_DOXY^2 + C_3 . CAL\_PHASE\_DOXY^3 + C_4 . CAL\_PHASE\_DOXY^4
$$

, where C$_{0...4}$ are temperature-dependent coefficients given on the calibration sheet as *C0-4 Coefficients* (temperature in $^{\circ}$C):

$$
C_i = C_{i0} + C_{i1} . TEMP + C_{i2} . TEMP^2 + C_{i3} . TEMP^3
$$

`MOLAR_DOXY` is in $\mu$mol L$^{-1}$ and represents the oxygen concentration at salinity = 0.

Default configuration:

```yaml
parameters:
    method: "POLY38"
    temperature_name: "TEMP"
    C0_coefs: [0, 0, 0, 0]
    C1_coefs: [0, 0, 0, 0]
    C2_coefs: [0, 0, 0, 0]
    C3_coefs: [0, 0, 0, 0]
    C4_coefs: [0, 0, 0, 0]
```

### Modified Stern-Volmer equation for pO$_2$

*This method has not yet been implemented.*


## 6.2 Aanderaa 4330 optodes

In contrast to the Aanderaa 3830 optodes, the conversion of `CAL_PHASE_DOXY` to `MOLAR_DOXY` for Aanderaa 4330 optodes can use the temperature values of the optode's temperature sensor (`TEMP_DOXY_SHIFTED`, shifted to geometrically align with CTD), as its accuracy and response time are much better than the older versions.

Default configuration:

```yaml
parameters:
    temperature_name: "TEMP_DOXY_SHIFTED"
```

### Stern-Volmer - SVU

This method uses the calibration coefficients derived from the multi-point calibration done that characterises the O$_2$-T dependence.

The temperature-dependent coefficients of this method are

$$ K_{SV} = c_0+c_1T+c_2T^2 $$

and

$$ P_0=c_3+c_4T $$

while the phase-dependent term is

$$ P_c=c_5+c_6 . CAL\_PHASE\_DOXY. $$

The freshwater-equivalent oxygen concentration is then:

$$ MOLAR\_DOXY_{S=0} = \frac{ P_0/P_c-1 }{ K_{SV} } $$

Calibration coefficents c$_{0...6}$ are provided in the manufacturer's calibration certificate as *SUVFoilCoefs*.

An optional two-point concentration adjustment may subsequently be applied:

$$ MOLAR\_DOXY_{S=0,adj} = MOLAR\_DOXY_{S=0} \cdot \mathrm{ConcCoef1} + \mathrm{ConcCoef0}. $$

The coefficients for this adjustment are given in the calibration certificate as *ConcCoefs*. The two-point adjustment is usually redundant if the multi-point calibration has been done for each individual sensor. However, in some cases like when calibrations have only been done at the batch-level, this correction is another opportunity to correct sensor-specific offsets to reference calibrations. If the coefficients are given by the manufacturer, this correction should be applied.

`MOLAR_DOXY` is in $\mu$mol L$^{-1}$ and represents the oxygen concentration at salinity = 0.

Default configuration:

```yaml
parameters:
    method: "SVU43"
    temperature_name: "TEMP_DOXY_SHIFTED"
    suvfoil_coefs: [0, 1, 0, 0, 0, 0]
    conc_coefs: [0, 1]
```

<font color=orange> I have changed the method name, but this has not been updated in pelagos yet. Have also added additional coefficient names but haven't yet checked how these have been implemented. </font>

### Polynomial

The polynomial method for the Aanderaa 4330 series calculates `MOLAR_DOXY` based on oxygen solubility (C$^{\ast}$; cm$^3$ dm$^{-3}$) and air saturation (AirSat; $\%$):

$$
MOLAR\_DOXY = C^{\ast} . 44.614 . \frac{AirSat}{100}
$$

The air saturation of oxygen is dependent on the partial pressure of O$_2$ (pO$_2$), which is determined from `CAL_PHASE_DOXY` using a two-dimensional polynomial:

$$
pO_2 =  C_0 . T^{m0} . CAL\_PHASE\_DOXY^{n0} + C_1 . T^{m1} . CAL\_PHASE\_DOXY^{m1} + ... + C_{27} . T^{m27} . CAL\_PHASE\_DOXY^{m27}
$$

, where C$_{0...13}$ and C$_{14...27}$ are provided on the calibration certificate as *FoilCoefA$_{0...13}$* and *FoilCoefB$_{0...13}$*, respectively. Temperature exponents m0...27 are shown as *FoilPolyDegT$_{0...27}$* and phase exponents n0...27 as *FoilPolyDegO$_{0...27}$*.

A two-point concentration adjustment may subsequently be applied (see above).

`MOLAR_DOXY` is in $\mu$mol L$^{-1}$ and represents the oxygen concentration at salinity = 0.

Similarly to the Stern-Volmer method, a two-point concentration adjustment can subsequently be applied if coefficients have been provided by the manufacturer (see above).

Default configuration:

```yaml
parameters:
    method: "POLY43"
    temperature_name: "TEMP_DOXY_SHIFTED"
    foil_a0_coefs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    foil_b0_coefs: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    foilpolydeg_T_coefs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    foilpolydeg_O_coefs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    conc_coefs: [0, 1]
```

### Modified Stern-Volmer equation for pO$_2$

*This method has not yet been implemented.*

***

# 7. Salinity correction

The optode measurements themselves are not dependent on salinity, as the sensor responds to the oxygen partial pressure ($pO{_2}$), which is independent of salinity. However, the O$_2$ concentration it measures is salinity-dependent, as it affects seawater O$_2$ solubility. The oxygen concentration determined using factory-provided calibration metadata is representative for freshwater, because it does not take this salinity-dependence into account. A correction is therefore necessary, which relates the the molar oxygen concentration for salinity = 0 to the molar oxygen concentration at seawater salinity (Bittig et al., 2016; 2018). The salinity-correction has two components. One is the correction term (S$_{corr}$), which addresses the effect of salinity on O$_2$ solubility. The other component adjusts the water vapour pressure (Pvapour$_{corr}$), which depends on salinity as it changes the atmospheric composition and therefore pO$_2$.

The overall salinity correction applied is represented by the following function:

$$
MOLAR\_DOXY_{PSAL} = Pvapour_{corr} . S_{corr} . MOLAR\_DOXY_{S=0}
$$

where:

- $S$ = practical salinity (`PRAC_SALINITY`)
- $T$ = temperature in $^{\circ}$C (`TEMP` from CTD for both Aanderaa 3830 optodes and Aanderaa 4330 optodes)
- $Pvapour_{\mathrm{corr}}$ = water-vapour partial pressure following Weiss & Price (1980)
- $S_{\mathrm{corr}}$ = salinity-dependent oxygen-solubility correction

The salinity correction factor (S$_{corr}$) is calculated as

$$S_{\mathrm{Corr}}=\exp\left[(S-S_{\mathrm{ref}})(B_0+B_1T_s+B_2T_s^2+B_3T_s^3)+C_0(S^2-S_{\mathrm{ref}}^2)\right]$$

where

$$T_s=\ln\left(\frac{298.15-T}{273.15+T}\right)$$

and $B_0,\ldots,B_3$ and $C_0$ are the coefficients recommended by the SCOR Working Group 142 (Bittig et al., 2016). S$_{ref}$ is reference salinity given in the optode settings and what is used my its internal calculations. This is normally equal to 0.


The water vapour pressure adjustment component (Pvapour$_{corr}$) is calculated as

$$
Pvapour_{corr} = \frac{1013.25-p_{\mathrm{H_2O}}(T,S_{preset})}{1013.25-p_{\mathrm{H_2O}}(T,S)}
$$

Water vapour pressure is calculated according to Weiss & Price (1980). S$_{preset}$ is set to 0.

Keep S$_{ref}$ and S$_{preset}$ = 0, as the former should not be used for `MOLAR_DOXY` re-calculated externally to the sensor and the latter needs to be set to 0 to account for the water vapour pressure effect, which Aanderaa neglects.

Default configuration:

```yaml
parameters:
      salinity_name: "PRAC_SALINITY"
      temperature_name: "TEMP"
      s_ref: 0
      s_preset: 0
```

---

# 8. Pressure correction on molar oxygen concentration

The pressure effect on luminescence lifetime needs to be applied to the uncalibrated phase shift. The pressure effect on quenching and on membrane O$_2$ equilibrium level needs to be applied to the oxygen concentration once it has been corrected for salinity (Bittig et al., 2018).

The pressure correction on molar oxygen concentration is applied:

$$\mathrm{MOLAR\_DOXY}_{\mathrm{PSAL,PCORR}}=\mathrm{MOLAR\_DOXY}_{\mathrm{PSAL}}\left[1+\left(P_{\mathrm{coef2}}T+P_{\mathrm{coef3}}\right)\frac{P}{1000}\right]$$

where:

- $T$ = CTD temperature in $^\circ$C (`TEMP`)
- $P$ = pressure in dbar (`PRES`)
- $P_{\mathrm{coef2}}$ and $P_{\mathrm{coef3}}$ = pressure-response coefficients

The correction should use the **CTD temperature**, not the optode temperature, regardless of the optode type.

The values of the pressure-response coefficients depends on whether the first pressure correction has been applied on the uncalibrated phase shift. For optodes where phase has already been corrected for pressure:

$$
P_{\mathrm{coef2}}=0.00022
$$

$$
P_{\mathrm{coef3}}=0.0419
$$

For optodes where phase has not been corrected for pressure:

$$
P_{\mathrm{coef2}}=0.00025
$$

$$
P_{\mathrm{coef3}}=0.0328
$$

These coefficients are set automatically by setting `uncalibrated_phase_correction_applied` in the configuration file to true or false. <font color=orange> If the first pressure correction step has been applied, pelagos-py will automatically set `uncalibrated_phase_correction_applied` to `true`.</font>

Default configuration:

```yaml
parameters:
      pressure_name: "PRES"
      temperature_name: "TEMP"
      molar_doxy_name: "MOLAR_DOXY_PSAL"
      uncalibrated_phase_correction_applied: true
```

---

# 9. Dissolved oxygen content
At this point, the oxygen values need to be QC'ed before proceeding to the remaining part of the pipeline. In order to apply the global and regional range checks, oxygen concentration needs to be expressed in units $\mu$mol kg$^{-1}$, i.e., the same units as the values given in the range checks. For all unit conversions, pelagos-py follows the best practice recommendations by SCOR WG 142 (Bittig et al., 2016). If the phase shift was converted to molar oxygen concentration in $\mu$mol L$^{-1}$ (including a subsequent salinity-correction), then the oxygen content is calculated using seawater density ($\rho$):


$$DOXY=\frac{MOLAR\_DOXY\_PSAL\_PCORR}{\rho}$$

Density is calculated using practical salinity and referenced to 0 dbar.

If the phase shift was converted to oxygen partial pressure in $\mu$atm, then the oxygen content is calculated as:

*Not yet implemented*

---

# 10. Oxygen QC

All QC checks documented here need to be applied to the oxygen content in $\mu$mol kg$^{-1}$, i.e., on the `DOXY` variable.

## 10.1 Global range check

Values outside physically plausible global ranges should be flagged as "bad data" (i.e., QC flag 4):

$$-5 \leq DOXY \leq 600 \quad[\mu\mathrm{mol\,kg^{-1}}]$$

Default configuration:

```yaml
parameters:
      qc_settings:
        range qc:
          variable_ranges:
            MOLAR_DOXY:
              4: [-5, 600, outside]
```

## 10.2 Spike check

A spike check is applied using a test value:

Test value = | V2 - (V3 + V1)/2 | - | (V3 - V1) / 2 |

where V2 is the measurement being tested as a spike, V1 and V3 are the values above and below.

**DOXY:** V2 value should be flagged as 4 "bad data", when:
- Test value > 50 µmol kg⁻¹ for pressure < 500 dbar
- Test value > 25 µmol kg⁻¹ for pressure ≥ 500 dbar


## 10.3 Gradient test

A gradient check is applied using a test value:

Test value = | V2 − (V3 + V1)/2 |

where V2 is the measurement being tested as a spike, and V1 and V3 are the values above and below.

**For DOXY:** the V2 value is flagged when:
- Test value exceeds 50 µmol/kg for pressures < 500 dbar
- Test value exceeds 25 µmol/kg for pressures ≥ 500 dbar


---

# 11. Response correction

*Still to be implemented.*

The optode cannot respond instantaneously to changes in ambient oxygen levels. This is seen most clearly when the sensor moves through large gradients in the water column when the oxygen profile looks smeared. It is also seen in profiles of down- and up-cast pairs, which will show a pattern that looks like a hysteresis effect. This reponse delay can lead to significant under- or over-estimations. The reason for the slow response is that oxygen needs to diffuse through the sensing foil in order to reach equilibrium with the ambient seawater. The diffusion is temperature- and solubility- dependent, and is also affected by the thickness of the liquid boundary layer above the membrane and flow of water. Bittig et al. (2014) have developed a 1-D diffusion model that describes the sensor's delay in response time (Bittig et al., 2018). However, scientific groups and communities working with glider optode data have developed other practical methods to correct for the response delay. Currently, no consensus exists on which method is "best" (López-García et al., 2022). pelagos-py therefore offers the following four most-used methods and provides diagnostics to compare, allowing the user to decide which one leads to the best results for their data. These methods have been adapted from López-García et al. (2022) to match the pelagos-py pipeline.

## 11.1 GEOMAR method


## 11.2 IMOS method


## 11.3 UEA method


## 11.4 AlterEco method



---

# 12. In-air calibration

*Not yet implemented*

---

# 13. Two-point calibration

*Not yet implemented*


***

# References

- Aanderaa (2007). TD 218 Operating Manual Oxygen Optodes.
- Bittig, H. C. et al. (2015). *Pressure Response of Aanderaa and Sea-Bird Oxygen Optodes*. Journal of Atmospheric and Oceanic Technology.
- Bittig, H. C. et al. (2016). SCOR Working Group 142 recommendations for oxygen quantity conversions.
- Bittig, H. C. et al. (2018). *Oxygen Optode Sensors: Principle, Characterization, Calibration, and Application in the Ocean*. Frontiers in Marine Science, 4, 429. https://doi.org/10.3389/fmars.2017.00429
- Garcia, H. E. & Gordon, L. I. (1992). Oxygen solubility in seawater: better fitting equations.
- Hahn, J. et al. — response-time characterization and correction of oxygen optodes.
- Uchida, H. et al. (2008). Oxygen optode calibration and temperature compensation.
- López-García,, P., Hull, T., Thomsen, S., Hahn, J., Queste, B.Y, et al (2022) OceanGliders Oxygen SOP, Version 1.0.0. OceanGliders, 55pp. DOI: http://dx.doi.org/10.25607/OBP-1756. (GitHub Repository, OceanGliders Oxygen SOP. Available: https://oceangliderscommunity.github.io/Oxygen_SOP/sections/authors_SOP_development_process.html.)