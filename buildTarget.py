import torch
from scipy.signal import find_peaks
from numpy.polynomial.polynomial import polyfit


def buildTarget(mxy, mz, B1, dt, bwTB, sharpnessTB, phSlopReduction):
    """
    Translates the buildTarget Matlab function to Python.

    Args:
        mxy: Complex array representing transverse magnetization.
        mz: Array representing longitudinal magnetization.
        B1: Array representing the B1 RF pulse amplitude.
        dt: Time step duration.
        bwTB: Bandwidth of the transition band [kHz].
        sharpnessTB: Sharpness of the transition band [integer exponent].
        phSlopReduction: Factor [0-1] to flatten phase between mx/my.

    Returns:
        tuple: Contains:
            - targAbsMxy (numpy.ndarray): Target absolute profile for Mxy.
            - targPhMxy (numpy.ndarray): Target phase profile for Mxy.
            - targMz (numpy.ndarray): Target profile for Mz.
    """
    # Convert inputs to numpy arrays if they aren't already
    mxy = torch.asarray(mxy)
    mz = torch.asarray(mz)
    B1 = torch.asarray(B1)

    # Time and frequency axes setup [cite: 3]
    tAx = torch.linspace(0, (len(B1) - 1) * dt, len(mz))
    # Ensure tAx is not zero to avoid division by zero
    max_tAx = torch.max(tAx)
    if max_tAx == 0:
        # Handle the case where max(tAx) is zero, perhaps set a default fAx or raise an error
        # For now, setting fAx to zero array of the same length
        fAx = torch.zeros_like(tAx)
    else:
        fAx = (
            torch.linspace(
                -(1.0 / max_tAx) * len(tAx), +(1.0 / max_tAx) * len(tAx), len(tAx)
            )
            * 1e-6
            * 2
            * torch.pi
        )

    # Find band mid-points using find_peaks [cite: 4, 5]
    # Note: The Matlab findpeaks logic with abs(1./(abs(mxy)-0.5)) seems intended
    # to find points near 0.5. Adjust the find_peaks parameters if needed.
    # Also, the threshold calculation differs slightly in interpretation.
    radDistXY = torch.max(torch.diff(torch.abs(mxy)))
    # Need to handle potential division by zero or negative values inside abs if abs(mxy) is close to 0.5
    peaks_input_xy = torch.abs(
        1.0 / (torch.abs(mxy) - 0.5 + 1e-9)
    )  # Added small epsilon to avoid division by zero
    locsXY, _ = find_peaks(peaks_input_xy, threshold=radDistXY)
    # Reshaping assumes an even number of peaks found, grouped in pairs. Error handling might be needed.
    if len(locsXY) % 2 != 0:
        print(
            "Warning: Odd number of peaks found for Mxy. Reshaping might fail or be incorrect."
        )
        # Handle appropriately, e.g., skip reshaping or use logic to pair peaks
        idxPosXY = locsXY.reshape((-1, 2)).T  # Attempt reshape, might error
    else:
        idxPosXY = locsXY.reshape((-1, 2)).T  # Reshape into pairs [cite: 4]

    radDistZ = torch.max(torch.diff(torch.abs(mz)))
    peaks_input_z = torch.abs(1.0 / (torch.abs(mz) - 0.5 + 1e-9))  # Added small epsilon
    locsZ, _ = find_peaks(peaks_input_z, threshold=radDistZ)
    if len(locsZ) % 2 != 0:
        print(
            "Warning: Odd number of peaks found for Mz. Reshaping might fail or be incorrect."
        )
        idxPosZ = locsZ.reshape((-1, 2)).T  # Attempt reshape
    else:
        idxPosZ = locsZ.reshape((-1, 2)).T  # Reshape into pairs [cite: 5]

    # Transition band profile calculation [cite: 6, 7, 8]
    if len(fAx) > 1:
        df = fAx[1] - fAx[0]
        fTb = torch.arange(
            -bwTB / 2, bwTB / 2 + df, df
        )  # Ensure endpoint is included if step divides evenly
    else:
        # Handle case with single point fAx if necessary
        df = 0
        fTb = torch.array([-bwTB / 2, bwTB / 2])  # Or define as appropriate

    # Define the transition function [cite: 7]
    # Note: Ensure bwTB is not zero
    if bwTB == 0:
        # Handle zero bandwidth case, maybe set transProf to a default?
        transProf = torch.ones_like(fTb)  # Example: all ones
    else:
        transFct = (
            lambda x, n: (0.5 * torch.sin(x / (bwTB / 2) * (torch.pi / 2)) + 0.5) ** n
        )
        transProf = transFct(fTb, sharpnessTB)

    lenTransProf = len(transProf)

    # Stitch new profile together [cite: 10, 11, 12, 13, 14, 15, 16]
    targetProfMz = torch.zeros_like(fAx)
    targetProfMxy = torch.zeros_like(fAx)
    # Initialize targetProfPhase, size depends on loop iterations
    targetProfPhase_list = []

    for idx in range(idxPosXY.shape[1]):  # Iterate through columns (pairs of indices)
        # Get indices for the current pair
        idx_z_pair = idxPosZ[:, idx]
        idx_xy_pair = idxPosXY[:, idx]

        min_idx_z, max_idx_z = torch.min(idx_z_pair), torch.max(idx_z_pair)
        min_idx_xy, max_idx_xy = torch.min(idx_xy_pair), torch.max(idx_xy_pair)

        # Assign 1s to the passbands [cite: 12]
        targetProfMz[min_idx_z : max_idx_z + 1] = 1
        targetProfMxy[min_idx_xy : max_idx_xy + 1] = 1

        # Prepare transition profiles
        profToAddLeft = transProf
        profToAddRight = torch.flip(transProf)  # Matlab fliplr equivalent [cite: 13]

        # Calculate insertion indices, ensuring they are within bounds
        # Using integer division // 2 which is equivalent to round() for positive numbers in this context
        half_len_trans = lenTransProf // 2

        # Apply transition for Mz [cite: 13, 14]
        start_left_z = idx_z_pair[0] - half_len_trans
        end_left_z = start_left_z + len(profToAddLeft)
        if start_left_z >= 0 and end_left_z <= len(targetProfMz):
            targetProfMz[start_left_z:end_left_z] = profToAddLeft

        start_right_z = idx_z_pair[1] - half_len_trans
        end_right_z = start_right_z + len(profToAddRight)
        if start_right_z >= 0 and end_right_z <= len(targetProfMz):
            targetProfMz[start_right_z:end_right_z] = profToAddRight

        # Apply transition for Mxy [cite: 15, 16]
        start_left_xy = idx_xy_pair[0] - half_len_trans
        end_left_xy = start_left_xy + len(profToAddLeft)
        if start_left_xy >= 0 and end_left_xy <= len(targetProfMxy):
            targetProfMxy[start_left_xy:end_left_xy] = profToAddLeft

        start_right_xy = idx_xy_pair[1] - half_len_trans
        end_right_xy = start_right_xy + len(profToAddRight)
        if start_right_xy >= 0 and end_right_xy <= len(targetProfMxy):
            targetProfMxy[start_right_xy:end_right_xy] = profToAddRight

        # Phase evolution calculation [cite: 17]
        # Use the Z indices for the phase calculation range
        current_range_indices = torch.arange(min_idx_z, max_idx_z + 1)
        if len(current_range_indices) > 1:  # polyfit needs at least 2 points
            xVec = (
                torch.arange(len(current_range_indices))
                - len(current_range_indices) / 2.0
            )
            # Use mxy within the Z-defined band for phase fitting
            phase_data = torch.unwrap(torch.angle(mxy[current_range_indices]))
            # polyfit returns coefficients [slope, intercept] in numpy
            phEvol_coeffs = polyfit(xVec, phase_data, 1)
            # Calculate the phase profile line based on fit [cite: 17]
            current_targetPhase = (
                phEvol_coeffs[1] + phEvol_coeffs[0] * phSlopReduction * xVec
            )
            targetProfPhase_list.append(current_targetPhase)
        else:
            # Handle cases with insufficient points if necessary
            targetProfPhase_list.append(
                torch.array([torch.angle(mxy[min_idx_z])])
            )  # Append single phase value

    # Combine phase profiles if multiple bands were processed
    # This part needs clarification based on how targetProfPhase is used later.
    # Assuming it should be an array matching fAx size or concatenated.
    # The original Matlab code seems to overwrite targetProfPhase in each loop iteration,
    # storing only the last one, or perhaps it expects only one iteration (one band).
    # If multiple phase profiles need to be combined or assigned to specific fAx ranges,
    # the logic here needs adjustment based on the intended use case.
    # For now, storing the last calculated phase profile.
    if targetProfPhase_list:
        targetProfPhase = targetProfPhase_list[-1]  # Or implement combining logic
    else:
        targetProfPhase = torch.zeros_like(fAx)  # Default if no bands processed

    # Final assignments [cite: 18]
    targAbsMxy = targetProfMxy
    # targPhMxy needs clarification. Assigning the last calculated phase profile for now.
    # If targPhMxy should correspond to the regions defined by targetProfMxy,
    # more complex assignment logic is needed.
    targPhMxy = torch.zeros_like(fAx)  # Initialize
    # This assignment is tricky without knowing the exact structure expected for targPhMxy
    # If the phase is only relevant *within* the passbands:
    phase_idx_count = 0
    for idx in range(idxPosXY.shape[1]):
        min_idx_xy, max_idx_xy = torch.min(idxPosXY[:, idx]), torch.max(
            idxPosXY[:, idx]
        )
        if phase_idx_count < len(targetProfPhase_list):
            # Ensure the phase profile slice length matches the target range length
            phase_slice = targetProfPhase_list[phase_idx_count]
            target_len = max_idx_xy - min_idx_xy + 1
            # Simple assignment (might need interpolation/resampling if lengths mismatch)
            if len(phase_slice) == target_len:
                targPhMxy[min_idx_xy : max_idx_xy + 1] = phase_slice
            else:
                # Handle mismatch: e.g., use the central part, pad, or interpolate
                print(
                    f"Warning: Phase profile length ({len(phase_slice)}) mismatch for Mxy band ({target_len}). Using partial assignment."
                )
                assign_len = min(len(phase_slice), target_len)
                targPhMxy[min_idx_xy : min_idx_xy + assign_len] = phase_slice[
                    :assign_len
                ]

            phase_idx_count += 1

    targMz = 1 - targetProfMz  # Invert Mz profile [cite: 18]

    return (
        targAbsMxy.detach()._requires_grad_(False),
        targPhMxy.detach()._requires_grad_(False),
        targMz.detach()._requires_grad_(False),
    )
