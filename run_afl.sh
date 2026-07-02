cargo afl build --release --package hpvcd-afl

AFL_SKIP_CPUFREQ=1 \
AFL_FAST_CAL=1 \
AFL_NO_STARTUP_CALIBRATION=1 \
AFL_CMPLOG_ONLY_NEW=1 \
cargo afl fuzz \
  -i ./corpus \
  -o ./findings \
  -t 1000+ \
  -m none \
  -c - \
  -- ./target/release/hpvcd-afl