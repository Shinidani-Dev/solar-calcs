from src.image import SolarImage


def main():
    # FITS
    img_fits = SolarImage.from_path("data/_10_03_54Z_SOLEX_82_530_N_SKY-82ED_ZWO-AZ183_10M-GM3000_Ha_20260205_100354.fits")
    img_fits.set_disk_from_header()

    sol_img_jpg = SolarImage.from_path("data/_10_03_54Z_SOLEX_82_530_N_SKY-82ED_ZWO-AZ183_10M-GM3000_Ha_20260205_100354_protus.jpg")
    sol_img_jpg.set_disk_from(img_fits)

    sol_img_jpg.bilateral_filter(5, 150, 150)
    sol_img_jpg.prominences_otsu(10, 180)
    mask = sol_img_jpg.copy().prominences_otsu(5, 50)  # jetzt ist mask.data binär
    profil = mask.calc_protuberanzenprofil(start_offset=0, max_length=50)

    sol_img_jpg.show_history()

    print(len(profil))
    for i in range(len(profil)):
        if profil[i] > 0:
            print(f"{i}: {profil[i]}")


if __name__ == "__main__":
    main()
