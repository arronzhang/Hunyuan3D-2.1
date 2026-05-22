PYTHON_BIN="${PYTHON:-python}"
EXT_SUFFIX="$("$PYTHON_BIN" - <<'PY'
import sysconfig

print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"

c++ -O3 -Wall -shared -std=c++11 -fPIC `$PYTHON_BIN -m pybind11 --includes` mesh_inpaint_processor.cpp -o "mesh_inpaint_processor${EXT_SUFFIX}"
