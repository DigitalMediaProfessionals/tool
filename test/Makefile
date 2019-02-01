include ../../env.mk

TARGETS = CaffeGoogLeNet/test \
					CaffeMobileNet/test \
					CaffeSqueezeNet/test \
					KerasGoogLeNet/test \
					KerasMobileNet/test \
					KerasSqueezeNet/test \
					SegNetBasic/test \
					YOLOv3/test \


CXX = $(GPP)
CXXFLAGS = $(OPT) -Wall -std=c++11 -pthread -I../../application/common/include
LDFLAGS = -ldmpdv -L../../application/common/lib -ldv700_util

all: $(TARGETS)

%/test: test.cpp %/testcfg.h
	$(CXX) $(CXXFLAGS)  -I$* -o $@ $< $(shell find $* -name '*_gen.cpp') $(LDFLAGS)

clean:
	rm -fr $(TARGETS)
