#raw audio file 
# InputAudio=test.raw
InputAudio=$1
#output directory:
# OutputDir=/tmp/phnrec_lat
OutputDir=$2
#phoneme recognizer:
# PHNREC=PHN_HU_SPDAT_LCRC_N1500
PHNREC=PHN_CZ_SPDAT_LCRC_N1500

OutputFilename=$3

#####################################################################


mkdir -p  ${OutputDir}/lattice ${OutputDir}/htkout

phnrecdir=/etc/PhnRec


echo PhnRec directory set to: $phnrecdir

alias phnrec=$phnrecdir/phnrec
echo alias phnrec set to $phnrecdir/phnrec



#generate file with posteriors for lattice generation
# !!!!!!!!!!  in $PHNREC/config in [posteriors] section has to be set  !!!!!!!!
# softening_func=gmm_bypass 0 0 0
echo "Posterior generation .... "  #odkomentovane
# echo "$InputAudio ${OutputDir}/htkout/${OutputFilename}.lin" > ${OutputDir}/list.post
# phnrec -t post -c $PHNREC -l ${OutputDir}/list.post
#or just one file
# phnrec -c $PHNREC -i $InputAudio -t post -o ${OutputDir}/htkout/${OutputFilename}.lin
#one file but from wave and not raw
phnrec -c $phnrecdir/$PHNREC -w lin16 -i $InputAudio -t post -o ${OutputDir}/htkout/${OutputFilename}.lin


#create list of files for decoding
echo "${OutputDir}/htkout/${OutputFilename}.lin" > ${OutputDir}/${OutputFilename}.scp

#lattice decoding
echo "Lattice decoding .... "
HVite \
-T 1 -y 'txt' -z 'latt'   \
-C ${OutputDir}/scoring/HVite.cfg   \
-w ${OutputDir}/scoring/monophones_lnet.hvite \
-n 2 1  \
-s 0 -p -1 \
-S ${OutputDir}/${OutputFilename}.scp \
-l ${OutputDir}/lattice   \
-H ${OutputDir}/scoring/hmmdefs.hvite \
${OutputDir}/scoring/dict \
${OutputDir}/scoring/hmmlist

rm ${OutputDir}/*.scp
rm ${OutputDir}/htkout/*.lin

#and lattices are in ${OutputDir}/lattice
   