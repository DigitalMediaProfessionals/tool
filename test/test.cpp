/*
 *  Copyright 2019 Digital Media Professionals Inc.

 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at

 *      http://www.apache.org/licenses/LICENSE-2.0

 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <cerrno>

#include "testcfg.h"

#define STR(s) #s
#define _STR(s) STR(s)
#define __NET_CLASS(name) C ## name
#define _NET_CLASS(X) __NET_CLASS(X)
#define NET_CLASS _NET_CLASS(NAME)
#define WEIGHT_FILE (_STR(NAME) "_weights.bin")
#define INPUT_MEM_LEN (PROC_W * PROC_H * PROC_C * PROC_D)

namespace test
{
	void test_zero(__fp16 *buf, size_t width, size_t height, size_t channel, size_t depth) 
	{
		memset(buf, 0, sizeof(*buf) * width * height * channel * depth);
	}
	void test_one(__fp16 *buf, size_t width, size_t height, size_t channel, size_t depth)
	{
		size_t len = width * height * channel * depth;
		for (size_t i = 0; i < len; i++) {
			buf[i] = 1.0;
		}
	}
	void test_max(__fp16 *buf, size_t width, size_t height, size_t channel, size_t depth)
	{
		memset(buf, 0xff, sizeof(*buf) * width * height * channel * depth);
	}
	void test_rand01(__fp16 *buf, size_t width, size_t height, size_t channel, size_t depth)
	{
		size_t len = width * height * channel * depth;
		for (size_t i = 0; i < len; i++) {
			buf[i] = static_cast<__fp16>(drand48());
		}
	}

	using input_generator = void (*)(__fp16 *buf, size_t width, size_t height, size_t channel, size_t depth);
	struct TestConfig
	{
		input_generator gen;
		std::string name;
	};

	TestConfig configs[] = {
		{.gen = test_zero, .name = "test_zero"},
		{.gen = test_one, .name = "test_one"},
		{.gen = test_max, .name = "test_max"},
		{.gen = test_rand01, .name = "test_rand01"},
	};

  void _mkrefdir()
  {
    int ret = mkdir("refs", 0777);
    if (ret == -1 && errno != EEXIST) {
      std::string msg = std::string("Failed to create refs/: ")
                        + std::string(std::strerror(errno));
      throw std::runtime_error(msg);
    }
  }

	void save_reference_output(TestConfig &cfg, int i_output, std::vector<float> &buf)
	{
		std::string path = std::string("refs/")
							+ cfg.name
							+ std::string("_")
							+ std::to_string(i_output);
    _mkrefdir();
		std::ofstream outf(path, std::ios_base::trunc | std::ios_base::binary);
		outf.exceptions(std::ios_base::badbit | std::ios_base::failbit);
		outf.write(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
    outf.close();
	}

	void load_reference_output(TestConfig &cfg, int i_output, std::vector<float> &buf)
	{
		std::string path = std::string("refs/")
							+ cfg.name
							+ std::string("_")
							+ std::to_string(i_output);
		std::ifstream inf(path, std::ios_base::in | std::ios_base::binary);
		inf.exceptions(std::ios_base::badbit | std::ios_base::failbit);
    inf.seekg(0, std::ios_base::end);
		size_t size = inf.tellg();
    inf.seekg(0, std::ios_base::beg);
    buf.resize(size / sizeof(float));
		inf.read(reinterpret_cast<char*>(buf.data()), size);
    inf.close();
	}

	class DMPException : public std::exception
	{
		public:
			DMPException(std::string msg)
			{
				_msg = msg;
			}
			DMPException(const char* msg)
			{
				_msg = std::string(msg);
			}
			const char* what() const noexcept
			{
				return _msg.c_str();
			}
		private:
			std::string _msg;
	};
};

namespace print
{
	using namespace std;
	const char default_color[]  = "\x1b[39m";
	const char green[]  = "\x1b[32m";
	const char yellow[] = "\x1b[33m";
	const char red[]    = "\x1b[31m";
	void pr_test_title(const test::TestConfig &cfg)
	{
		cout << endl;
		cout << "[" << cfg.name << "]" << endl;
	}

	void pr_result(bool ok)
	{
		cout << "<Result> - ";
		if (ok) {
			cout << green << "Passed" << default_color << endl;
		} else {
			cout << red << "Failed" << default_color << endl;
		}
    cout << endl;
	}

	void pr_global_title()
	{
		cout << "======== Test for " << _STR(NAME) << " ========" << endl;
	}

	void pr_global_result(unsigned num_ok, unsigned num_ng, unsigned num_exception)
	{
		cout << endl;
		cout << "[Overall Result]" << endl;
		cout << green << "\t # of Passed - " << num_ok << endl;
		cout << red << "\t # of Failed - " << num_ng << endl;
		cout << yellow << "\t # of Exception - " << num_exception << endl;
		cout << default_color << endl;
	}

  void usage()
  {
    cerr << "./test test|save_output\n"
      << "\t test: do test\n"
      << "\t save_output: save network output as a reference output\n"
      << endl;
  }
};

namespace 
{
	using namespace std;
	NET_CLASS network;
	void init()
	{
		network.Verbose(0);
		if (!network.Initialize()) {
			throw runtime_error("Failed to initialize network");
		}
		if (!network.LoadWeights(WEIGHT_FILE)) {
			throw runtime_error("Failed to load network weight");
		}
		if (!network.Commit()) {
			throw runtime_error("Failed to commit network");
		}

		srand48(41);
	}

	bool run(test::TestConfig cfg, bool save_output)
	{
		__fp16 *input_addr = reinterpret_cast<__fp16*>(network.get_network_input_addr_cpu());
		cfg.gen(input_addr, PROC_W, PROC_H, PROC_C, PROC_D);
		bool same_output = true;

		if(!network.RunNetwork()) {
			throw test::DMPException("Failed to run network");
		}
		for (int i = 0; i < network.get_output_layer_count(); i++) {
			// get output
			vector<float> output;
			network.get_final_output(output, i);
			if (save_output) {
				test::save_reference_output(cfg, i, output);
			} else { 
				// get reference output
        vector<float> ref_buf;
				test::load_reference_output(cfg, i, ref_buf);
				// compare output
				if (ref_buf != output) {
					same_output = false;
					cout << i << "-th output differs from the reference" << endl;
				}
			}
		}
		return same_output;
	}
};

int main(int argc, char const* argv[])
{
	unsigned num_ok = 0;
	unsigned num_ng = 0;
	unsigned num_exception = 0;

	init();
	bool save_output = (argc > 1 &&
						(strncmp(argv[1], "save_output", sizeof("save_output")) == 0));
	bool do_test = (argc > 1 && (strncmp(argv[1], "test", sizeof("test")) == 0));
	if (!(do_test || save_output)) {
    print::usage();
		return -1;
	}
	print::pr_global_title();
	for (auto& cfg: test::configs) {
		try {
			print::pr_test_title(cfg);
			bool ok = run(cfg, save_output);
			if (ok) {
				num_ok++;
			} else {
				num_ng++;
			}
			print::pr_result(ok);
		} catch (test::DMPException &e) {
			num_exception++;
			cerr << e.what() << endl;
		}
	}

	print::pr_global_result(num_ok, num_ng, num_exception);

  if (num_ng == 0 && num_exception == 0) {
    return -1;
  } else {
    return 0;
  }
}
