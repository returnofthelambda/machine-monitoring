# 📦 Enclosures, Cases & Industrial Hardware Protection

This guide covers physical enclosure options, environmental sealing, and PCB conformal coating for deploying edge sensors in harsh industrial environments.

---

## 1. Physical Enclosure Options

### A. 3D-Printable Custom Cases
* **Design Files:** Search [Printables](https://www.printables.com) or [Thingiverse](https://www.thingiverse.com) for `AmebaPro2 AMB82-MINI case` or `AMB82-MINI mount`.
* **Features:** Recommended designs include snap-fit lids, cutouts for the camera sensor, analog microphone port, USB-C power lead, and external IPEX antenna cable routing.
* **Filament:** Use **PETG or ABS** for industrial or mechanical equipment environments to prevent thermal warping.

### B. Industrial IP65 / IP67 Sealed Polycarbonate Junction Boxes
* For factory floors with particulate dust, water spray, or oil mist:
  * Standard ABS/Polycarbonate junction boxes ($100\times68\times50\text{ mm}$) with silicone O-ring gaskets and M12/PG7 cable glands.
  * **Acoustic Membrane Protection:** Place an adhesive **hydrophobic/acoustic vent membrane** (e.g. GORE acoustic vent sticker) over the microphone aperture hole. This allows acoustic sound waves to pass with minimal attenuation while maintaining an impermeable liquid and dust barrier.

---

## 2. SoC Resilience: AMB82-MINI vs Single Board Computers

The AMB82-MINI is significantly more resilient to harsh industrial settings than conventional SBCs (e.g. Raspberry Pi 4/5):

| Feature | Realtek RTL8735b (AMB82-MINI) | Conventional SBC (Raspberry Pi 4/5) |
|---|---|---|
| **Cooling & Airflow** | **100% Passive (< 1W power draw)** — can be sealed in an airtight, non-ventilated box with zero airflow. | Requires heatsinks, ventilation slots, or active fans that suck airborne industrial dust across components. |
| **Silicon Packaging** | **System-in-Package (SiP)** — CPU, NPU, DSP, Wi-Fi, and RAM are co-packaged in a single sealed BGA chip, eliminating exposed high-speed trace pins. | Discrete high-speed RAM and controller chips with exposed surface solder pins vulnerable to conductive dust bridging. |
| **Storage Medium** | **Solid-State NOR/SPI Flash** — completely resilient to vibration and mechanical wear. | Fragile MicroSD card slots prone to oxidation, dust contamination, or mechanical contact failure. |

---

## 3. Recommended Industrial PCB Hardening

For deployment in environments with conductive metal shavings, high humidity, or chemical vapors:
1. Mask the microphone pinhole and camera lens with Kapton tape.
2. Apply an **acrylic or silicone conformal coating spray** (e.g., MG Chemicals 419D) over both sides of the PCB.
3. Secure the external IPEX/U.FL antenna connector with a drop of neutral-cure silicone or hot glue to prevent vibration loosening.
