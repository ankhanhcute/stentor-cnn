import Pkg; Pkg.add("HDF5")
using HDF5

f = h5open("2025_09_12_01_33_36_contractions.h5", "r")
for key in keys(f)
    data = read(f[key])
    println("=== $key ===")
    println("  size: ", size(data))
    println("  type: ", typeof(data))
    if key == "manual"
        println("  1s (contracted): ", sum(data .== 1.0))
        println("  0s (not contracted): ", sum(data .== 0.0))
        println("  NaNs (skipped): ", sum(isnan.(data)))
    end
end

close(f)