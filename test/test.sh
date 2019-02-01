#!/bin/bash

gen_src() {
  echo "Generating network source code and weight"
  for ini in model/*.ini
  do
    python3 ../convertor.py $ini
  done
}

if [[ "$1" == "save_reference" ]]; then
  save_reference=true
elif [[ -z "$1" ]]; then
  :
else
  echo '[Usage]'
  echo './test.sh [save_reference]'
  echo '    save_reference: do not test but save output of networks as reference'
  echo
  exit -1
fi

exit_code=0
_PWD=$PWD
if $save_reference; then
  save_output=save_output
else
  # check if the generated files are created by the latest convertor
  latest_tool_srcs=$(git log --oneline --namy-only -1 | sed -n '2,$p')
  latest_src=""
  for s in latest_tool_srcs; do
    if [[ -z "$latest_src" ]]; then
      latest_src=$s
    elif [[ "$s" -nt "$latest_src" ]]; then
      latest_src=$s
    fi
  done

  must_update=false
  for f in $(find . -maxdepth 2 -name '*_gen.*') $(find . -maxdepth 2 -name '*_weights.bin'); do
    if [[ "$f" -ot "$latest_src" ]]; then
      must_update=true
      echo Update $f !
    fi
  done
  if $must_update; then
    exit -1
  fi
fi


subdirs=$(find . -maxdepth 1 -type d | egrep "\./" | grep -v model)
make -j 4
for d in $subdirs
do
  cd $_PWD/$d && ./test $save_output
  if [[ "$?" != "0" ]]; then
    exit_code=$?
  fi
done

exit $exit_code
