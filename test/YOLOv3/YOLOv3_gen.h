/*
 *  Copyright 2018 Digital Media Professionals Inc.

 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at

 *      http://www.apache.org/licenses/LICENSE-2.0

 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.

 *  This source code was generated using DMP-DV700 tools.
 */
#pragma once

#include "dmp_network.h"

class CYOLOv3 : public CDMP_Network {
 private:
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 0 | FPGA-Layer | Convolution | (576, 288, 3) | (576, 288, 16) | - | - |
  | 0-0 | conv2d_1 | Convolution | (576, 288, 3) | (576, 288, 16) | - | 2336 |

  */
  void Layer_0();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 1 | FPGA-Layer | Convolution | (576, 288, 16) | (288, 144, 16) | - | - |
  | 1-0 | max_pooling2d_1 | Pooling | (576, 288, 16) | (288, 144, 16) | - | - |

  */
  void Layer_1();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 2 | FPGA-Layer | Convolution | (288, 144, 16) | (288, 144, 32) | - | - |
  | 2-0 | conv2d_2 | Convolution | (288, 144, 16) | (288, 144, 32) | - | 9280 |

  */
  void Layer_2();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 3 | FPGA-Layer | Convolution | (288, 144, 32) | (144, 72, 32) | - | - |
  | 3-0 | max_pooling2d_2 | Pooling | (288, 144, 32) | (144, 72, 32) | - | - |

  */
  void Layer_3();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 4 | FPGA-Layer | Convolution | (144, 72, 32) | (144, 72, 64) | - | - |
  | 4-0 | conv2d_3 | Convolution | (144, 72, 32) | (144, 72, 64) | - | 36992 |

  */
  void Layer_4();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 5 | FPGA-Layer | Convolution | (144, 72, 64) | (72, 36, 64) | - | - |
  | 5-0 | max_pooling2d_3 | Pooling | (144, 72, 64) | (72, 36, 64) | - | - |

  */
  void Layer_5();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 6 | FPGA-Layer | Convolution | (72, 36, 64) | (36, 18, 128) | - | - |
  | 6-0 | conv2d_4 | Convolution | (72, 36, 64) | (72, 36, 128) | - | 147712 |
  | 6-0 | max_pooling2d_4 | Pooling | (72, 36, 128) | (36, 18, 128) | - | - |

  */
  void Layer_6();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 7 | FPGA-Layer | Convolution | (36, 18, 128) | (36, 18, 256) | - | - |
  | 7-0 | conv2d_5 | Convolution | (36, 18, 128) | (36, 18, 256) | - | 590336 |

  */
  void Layer_7();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 8 | FPGA-Layer | Convolution | (36, 18, 256) | (18, 9, 256) | - | - |
  | 8-0 | max_pooling2d_5 | Pooling | (36, 18, 256) | (18, 9, 256) | - | - |

  */
  void Layer_8();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 9 | FPGA-Layer | Convolution | (18, 9, 256) | (18, 9, 512) | - | - |
  | 9-0 | conv2d_6 | Convolution | (18, 9, 256) | (18, 9, 512) | - | 2360320 |
  | 9-0 | max_pooling2d_6 | Pooling | (18, 9, 512) | (18, 9, 512) | - | - |

  */
  void Layer_9();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 10 | FPGA-Layer | Convolution | (18, 9, 512) | (18, 9, 1024) | - | - |
  | 10-0 | conv2d_7 | Convolution | (18, 9, 512) | (18, 9, 1024) | - | 9439232 |

  */
  void Layer_10();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 11 | FPGA-Layer | Convolution | (18, 9, 1024) | (18, 9, 256) | - | - |
  | 11-0 | conv2d_8 | Convolution | (18, 9, 1024) | (18, 9, 256) | - | 590336 |

  */
  void Layer_11();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 12 | FPGA-Layer | Convolution | (18, 9, 256) | (18, 9, 512) | - | - |
  | 12-0 | conv2d_9 | Convolution | (18, 9, 256) | (18, 9, 512) | - | 2360320 |

  */
  void Layer_12();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 13 | FPGA-Layer | Convolution | (18, 9, 512) | (18, 9, 255) | - | - |
  | 13-0 | conv2d_10 | Convolution | (18, 9, 512) | (18, 9, 255) | - | 294272 |

  */
  void Layer_13();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 14 | FPGA-Layer | Flatten | (18, 9, 255) | (41310,) | - | - |

  */
  void Layer_14();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 15 | FPGA-Layer | Convolution | (18, 9, 256) | (18, 9, 128) | - | - |
  | 15-0 | conv2d_11 | Convolution | (18, 9, 256) | (18, 9, 128) | - | 73984 |

  */
  void Layer_15();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 16 | FPGA-Layer | Convolution | (18, 9, 128) | (36, 18, 128) | - | - |
  | 16-0 | up_sampling2d_1 | UpSampling | (18, 9, 128) | (36, 18, 128) | - | - |

  */
  void Layer_16();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 17 | FPGA-Layer | Concatenate | (36, 18, 384) | (36, 18, 384) | - | - |

  */
  void Layer_17();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 18 | FPGA-Layer | Convolution | (36, 18, 384) | (36, 18, 256) | - | - |
  | 18-0 | conv2d_12 | Convolution | (36, 18, 384) | (36, 18, 256) | - | 1769984 |

  */
  void Layer_18();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 19 | FPGA-Layer | Convolution | (36, 18, 256) | (36, 18, 255) | - | - |
  | 19-0 | conv2d_13 | Convolution | (36, 18, 256) | (36, 18, 255) | - | 147392 |

  */
  void Layer_19();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 20 | FPGA-Layer | Flatten | (36, 18, 255) | (165240,) | - | - |

  */
  void Layer_20();
  /*!

  Layer description

  | ID | Layers | Type | Dim In | Dim Out | Param | Mem |
  | :- | :- | :-: | :-: | :-: | :-: | :-: |
  | 21 | FPGA-Layer | Concatenate | (206550,) | (206550,) | - | - |

  */
  void Layer_21();

 public:
  virtual bool Initialize();
  CYOLOv3();
  virtual ~CYOLOv3();
};
