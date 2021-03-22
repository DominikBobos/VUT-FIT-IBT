OutputDir=$1
PHNREC=PHN_CZ_SPDAT_LCRC_N1500

#####################################################################

mkdir -p ${OutputDir}/scoring 
phnrecdir=/etc/PhnRec
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