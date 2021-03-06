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



mkdir -p ${OutputDir}/scoring ${OutputDir}/lattice ${OutputDir}/htkout

phnrecdir=/etc/PhnRec


echo PhnRec directory set to: $phnrecdir

alias phnrec=$phnrecdir/phnrec
echo alias phnrec set to $phnrecdir/phnrec

#initialization
echo "Initialization .... "
echo "TARGETKIND     = MFCC
ALLOWXWRDEXP   = T
" > ${OutputDir}/scoring/HVite.cfg

cat ${phnrecdir}/$PHNREC/dicts/phonemes | grep -vE "oth|spk|int" > ${OutputDir}/scoring/hmmlist
cat ${OutputDir}/scoring/hmmlist | awk '{print $1,$1}' > ${OutputDir}/scoring/dict
cat ${phnrecdir}/$PHNREC/dicts/phonemes | awk '{printf $1"__1\n"$1"__2\n"$1"__3\n" }' > ${OutputDir}/scoring/states

#create recognition net
HBuild ${OutputDir}/scoring/hmmlist ${OutputDir}/scoring/monophones_lnet.hvite

#HMM general
#do.HMM.sh ${OutputDir}/scoring/states ${OutputDir}/scoring/hmmdefs.hvite
#HMM for SPDAT (where phonemes int,spk,pau are merged together to form one phoneme pau)
./do.HMM_PauSpkInt.sh ${OutputDir}/scoring/states ${OutputDir}/scoring/hmmdefs.hvite


#generate file with posteriors for lattice generation
# !!!!!!!!!!  in $PHNREC/config in [posteriors] section has to be set  !!!!!!!!
# softening_func=gmm_bypass 0 0 0
echo "Posterior generation .... "
echo "$InputAudio ${OutputDir}/htkout/${OutputFilename}.lop" > ${OutputDir}/list.post
# phnrec -t post -c $PHNREC -l ${OutputDir}/list.post
#or just one file
# phnrec -c $PHNREC -i $InputAudio -t post -o ${OutputDir}/htkout/${OutputFilename}.lop
#one file but from wave and not raw
phnrec -c $phnrecdir/$PHNREC -w lin16 -i $InputAudio -t post -o ${OutputDir}/htkout/${OutputFilename}.lop


#create list of files for decoding
echo "${OutputDir}/htkout/${OutputFilename}.lop" > ${OutputDir}/${OutputFilename}.scp

#lattice decoding
echo "Lattice decoding .... "
HVite \
-T 1 -y 'rec' -z 'latt'   \
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

#and lattices are in ${OutputDir}/lattice
   