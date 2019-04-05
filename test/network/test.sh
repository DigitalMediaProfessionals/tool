#!/bin/bash

set -e

change_quant_transweight() {
  local ini=$1
  local quantization=$2
  local transweight=$3
  sed -i "s/quantization *= *[01]/quantization = $quantization/" $1
  sed -i "s/transpose_weight *= *[01]/transpose_weight = $transweight/" $1
}

usage() {
  echo "Usage: ./test.sh Options.."
  echo
  echo "Options:"
  echo "   --update-ref    update reference"
  echo "   --model model   target model"
  echo "   --verbose       enable verbose mode"
  echo
}

verbose_print() {
  if $verbose; then
    echo $@
  fi
}

print_cmp_result() {
  local RET=$1
  local gen=$2
  local ref=$3
  case "$RET" in
    "0")
      ;;
    "1")
      fail=true
      echo "[LOG] $gen_w and $ref_w differs"
      ;;
    *)
      fail=true
      echo Trouble on comparing $gen_w with $ref_w 1>&2
      ;;
  esac
}

update_ref=false
verbose=false
while [[ -n "$@" ]]
do
  case $1 in
    "--update-ref" )
      update_ref=true
      shift
      ;;
    "--verbose" )
      verbose=true
      shift
      ;;
    "--model" )
      model=$2
      shift 2
      ;;
    *)
      echo '[ERROR] Unknown arguments are passed'
      usage
      exit -1
      ;;
  esac
done


# initialize variables before the main loop
if [[ -n "$model" ]]
then
  models=$model
else
  models=$(cat model.list)
fi
num_succ=0
num_fail=0

for m in $models
do
  m=${m%/}
  for qt in "11" "01"
  do
    quant=${qt:0:1}
    transweight=${qt:1:1}
    ini=$m/$m.ini
    refd=$m/refs/q${quant}t${transweight}/

    # generate the source and weight
    verbose_print "Generating source and weight for $m"
    verbose_print "quantization: $quant"
    verbose_print "transpose_weight: $transweight"
    change_quant_transweight $ini $quant $transweight
    python3 ../../convertor.py $ini --cv2_seed 0

    gen_src=$m/${m}_gen.cpp
    gen_hdr=$m/${m}_gen.h
    gen_w=$m/${m}_weights.bin

    ref_src=$refd/${m}_gen.cpp
    ref_hdr=$refd/${m}_gen.h
    ref_w=$refd/${m}_weights.bin

    set +e
    if $update_ref; then
      # store the generated as a reference
      verbose_print "Storing reference for $m"
      mkdir -p $refd
      cp $gen_src $ref_src
      cp $gen_hdr $ref_hdr
      cp $gen_w $ref_w
    else
      # compare the generated and the reference
      verbose_print "Comparing the generated and the reference"
      fail=false
      cmp -s $gen_w $ref_w
      print_cmp_result $? $gen_w $ref_w
      diff $gen_src $ref_src >/dev/null
      print_cmp_result $? $gen_src $ref_src
      diff $gen_hdr $ref_hdr >/dev/null
      print_cmp_result $? $gen_hdr $ref_hdr

      # increment counter and output the result
      echo -n "[Result] $m with {quantization: $quant, transpose_weight: $transweight} : "
      if $fail; then
        echo "Failed"
        num_fail=$(( $num_fail + 1 ))
      else
        echo "Successed"
        num_succ=$(( $num_succ + 1 ))
      fi
    fi
    set -e
  done
done

if $update_ref; then
  :
else
  # show final result
  echo "[Result] Total tests - $(( $num_fail + $num_succ ))"
  echo "[Result] Successed - $num_succ"
  echo "[Result] Failed - $num_fail"

  if [[ "$num_fail" -gt 0 ]]; then
    exit -1
  else
    exit 0
  fi
fi
