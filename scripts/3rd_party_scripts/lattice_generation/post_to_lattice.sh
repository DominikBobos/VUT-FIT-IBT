InputPost=$1
OutputDir=$2
PHNREC=PHN_CZ_SPDAT_LCRC_N1500
OutputFilename=$3

#####################################################################

#create list of files for decoding
echo "${InputPost}.lin" > ${OutputDir}/${OutputFilename}.scp

#lattice decoding
echo "Lattice decoding .... "
HVite \
-T 1 -y 'txt' -z 'latt'   \
-C ${OutputDir}/scoring/HVite.cfg   \
-w ${OutputDir}/scoring/monophones_lnet.hvite \
-n 2 1  \
-s 0 -p -1 \
-S ${OutputDir}/${OutputFilename}.scp \
-l ${OutputDir}   \
-H ${OutputDir}/scoring/hmmdefs.hvite \
${OutputDir}/scoring/dict \
${OutputDir}/scoring/hmmlist

rm ${OutputDir}/*.scp

#and lattices are in ${OutputDir}/lattice
   