"""Training regions: rural areas in different parts of India where village
ponds and tanks are mapped in OpenStreetMap. Each is a (west, south, east,
north) box of about 0.3 x 0.3 degrees. Cross-validation holds out whole
regions, so the reported accuracy is for terrain the model has never seen.
"""

REGIONS = {
    "raipur_cg": (81.60, 21.00, 81.90, 21.30),       # Chhattisgarh plains
    "bemetara_cg": (81.30, 21.55, 81.60, 21.85),     # Chhattisgarh plains
    "medak_ts": (78.10, 17.90, 78.40, 18.20),        # Telangana tanks
    "kancheepuram_tn": (79.70, 12.60, 80.00, 12.90), # Tamil Nadu eris
    "kolar_ka": (78.00, 13.00, 78.30, 13.30),        # Karnataka tank cascades
    "alwar_rj": (76.40, 27.40, 76.70, 27.70),        # Rajasthan johads
    "sehore_mp": (77.00, 23.00, 77.30, 23.30),       # Malwa plateau
    "jalna_mh": (75.80, 19.80, 76.10, 20.10),        # Deccan plateau
    "bankura_wb": (87.00, 23.10, 87.30, 23.40),      # laterite uplands
    "ranchi_jh": (85.20, 23.20, 85.50, 23.50),       # Chota Nagpur plateau
}


# ---------------------------------------------------------------------------
# Chhattisgarh model (the app's target state). Fixed on 25 Oct 2026 before
# any of the new districts were looked at:
#   * TRAIN: districts used for development and cross-validation
#   * TEST:  held out completely; scored once, after all choices were made
# Raipur and Bemetara were also part of the all-India experiment, so they are
# training regions only.
# ---------------------------------------------------------------------------

CHHATTISGARH_TRAIN = {
    "raipur_cg": (81.60, 21.00, 81.90, 21.30),
    "bemetara_cg": (81.30, 21.55, 81.60, 21.85),
    "durg_cg": (81.20, 21.00, 81.50, 21.30),
    "dhamtari_cg": (81.40, 20.55, 81.70, 20.85),
    "balodabazar_cg": (82.00, 21.55, 82.30, 21.85),
    "kawardha_cg": (81.15, 21.90, 81.45, 22.20),
}

CHHATTISGARH_TEST = {
    "mahasamund_cg": (82.00, 21.00, 82.30, 21.30),
    "janjgir_cg": (82.40, 21.90, 82.70, 22.20),
    "rajnandgaon_cg": (80.80, 21.00, 81.10, 21.30),
}

# Area where the Chhattisgarh model is used by the app (else: formula).
CHHATTISGARH_BOUNDS = (80.2, 17.7, 84.4, 24.1)
