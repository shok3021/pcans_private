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

############# PSD calculation RESUME (Skip 26000) ###############
# 26000 (壊れ) を避けて、26500以降を拾うパターン
# 026[5-9]?? ... 26500, 26600, 26700, 26800, 26900 にマッチ
# 02[7-9]??? ... 27000〜29999 にマッチ
# 0[3-4]???? ... 30000〜49999 にマッチ
# 050000     ... 50000 にマッチ

PSD_FILES = $(sort $(wildcard /data/shok/dat/026[5-9]??_rank=*.dat \
                              /data/shok/dat/02[7-9]???_rank=*.dat \
                              /data/shok/dat/0[3-4]????_rank=*.dat \
                              /data/shok/dat/050000_rank=*.dat))

psd_calc_resume : $(LIB_DIR)/psd/psd.out
	$(LIB_DIR)/psd/psd.out 160 64 ./ $(PSD_FILES)
####################################################

clean :
	rm -f $(TARGET) $(OBJS) *.mod dat/*.dat mom/*.dat psd/*.dat

