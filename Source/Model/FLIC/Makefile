_flm.so: _flm.cpp
	@c++ -O3 -shared -std=c++11 -I pybind11/include `python-config --cflags --ldflags --libs` _flm.cpp -fPIC -Wall -pedantic -o _flm.so
