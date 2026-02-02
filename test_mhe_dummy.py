from infNMPC_options import _import_settings
from dummy_mhe_test_data import make_dummy_io_data
from MHE_estimation import solve_mhe_no_arrival_cost


if __name__ == "__main__":
    options = _import_settings()

    # Make sure your new option exists (after you add it)
    M_desired = getattr(options, "MHE_window", 5)

    # Fake io_data_array in the same format as run_MPC logs:
    # [Cc(k), T(k), Fa0(k-1), mc(k-1)] with row0 MVs = None
    io_data_array = make_dummy_io_data(num_steps=10)

    mhe = solve_mhe_no_arrival_cost(
        options=options,
        io_data_array=io_data_array,
        M_desired=M_desired,
        tee=True,
    )
    print("mhe is None?", mhe is None)
if mhe is not None:
    print("M_eff:", mhe.M_eff)
    print("xhat type:", type(mhe.xhat))
    print("len(xhat):", len(mhe.xhat))
    print("xhat keys:", sorted(list(mhe.xhat.keys()))[:20])
    print("xhat dict:", mhe.xhat)

    if mhe is None:
        print("MHE returned None (likely k<1 or M_eff<1).")
    else:
        print("M_eff:", mhe.M_eff)
        print("xhat at current time:")
        for k, v in sorted(mhe.xhat.items()):
            print(f"  {k}: {v}")
