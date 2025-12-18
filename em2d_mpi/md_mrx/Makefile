LIB_DIR = ..
include $(LIB_DIR)/Makefile_inc
TARGET = a.out
OBJS = const.o init.o main.o
.PHONY : all 
.PHONY : clean
.SUFFIXES :
.SUFFIXES : .o .f90
.f90.o:
	$(FC) -L$(LIB_DIR) $(FFLAGS) -I $(LIB_DIR)/common -c $< -lem2d

all: $(TARGET)

$(TARGET) : $(OBJS)
	$(FC) -L$(LIB_DIR) -o $(TARGET) $(FFLAGS) $(OBJS) -lem2d 

$(LIB_DIR)/libem2d.a : 
	cd $(LIB_DIR) ; make

main.o : init.o const.o $(LIB_DIR)/libem2d.a
init.o : const.o $(LIB_DIR)/libem2d.a

############# moment calculation ###############
moment : $(LIB_DIR)/moment/mom.out
	$(LIB_DIR)/moment/mom.out ./ ./dat/*_rank*.dat

$(LIB_DIR)/moment/mom.out :
	cd $(LIB_DIR)/moment ; make
###############################################

############# PSD calculation ###############
psd_calc : $(LIB_DIR)/psd/psd.out
	$(LIB_DIR)/psd/psd.out 160 64 ./ /data/shok/dat/*_rank*.dat

$(LIB_DIR)/psd/psd.out :
	cd $(LIB_DIR)/psd ; make
###############################################

############# PSD calculation RESUME ###############
# ファイル名パターン: 026000_rank=000.dat (6桁ゼロ埋め)
# 26000-29999 : 02[6-9]???...
# 30000-49999 : 0[3-4]????...
# 50000       : 050000...

PSD_FILES = $(sort $(wildcard /data/shok/dat/02[6-9]???_rank=*.dat \
                              /data/shok/dat/0[3-4]????_rank=*.dat \
                              /data/shok/dat/050000_rank=*.dat))

psd_calc_resume : $(LIB_DIR)/psd/psd.out
	$(LIB_DIR)/psd/psd.out 160 64 ./ $(PSD_FILES)
####################################################

clean :
	rm -f $(TARGET) $(OBJS) *.mod dat/*.dat mom/*.dat psd/*.dat

