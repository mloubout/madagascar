#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <rpc/xdr.h>

#include "file.h"
#include "c99.h"
#include "getpar.h"
#include "simtab.h"
#include "alloc.h"
#include "error.h"

struct sf_File {
    FILE *stream; 
    char *dataname, *buf;
    sf_simtab pars;
    XDR *xdr;
    enum xdr_op op;
    sf_datatype type;
    sf_dataform form;
};

/*@null@*/ static sf_file infile=NULL;
static const int tabsize=10;
/*@null@*/ static char *aformat = NULL;
static size_t aline=8;

static bool getfilename (FILE *fd, char *filename);
static char* getdatapath (void);
static bool readpathfile (const char* filename, char* datapath);

sf_file sf_input (/*@null@*/ char* tag)
{
    int esize;
    sf_file file;
    char *filename, *format;
    size_t len;

    file = (sf_file) sf_alloc(1,sizeof(*file));
    
    if (NULL == tag || 0 == strcmp(tag,"in")) {
	file->stream = stdin;
	filename = NULL;
    } else {
	filename = sf_getstring (tag);
	if (NULL == filename) {
	    len = strlen(tag)+1;
	    filename = sf_charalloc(len);
	    memcpy(filename,tag,len);
	}

	file->stream = fopen(filename,"r");
	if (NULL == file->stream) 
	    sf_error ("%s: Cannot read header file %s:",__FILE__,filename);
    }
/*    setbuf(file->stream,file->buf); */

    file->pars = sf_simtab_init (tabsize);
    sf_simtab_input (file->pars,file->stream);
    if (NULL == filename) {
	infile = file;
    } else {
	free (filename);
    }

    filename = sf_histstring(file,"in");
    if (NULL == filename) sf_error ("%s: No in= in file %s",__FILE__,tag);
    len = strlen(filename)+1;
    file->dataname = sf_charalloc(len);
    memcpy(file->dataname,filename,len);

    if (0 != strcmp(filename,"stdin")) {
	file->stream = freopen(filename,"rb",file->stream);
	if (NULL == file->stream) 
	    sf_error("%s: Cannot read data file %s:",__FILE__,filename);
    }
    free (filename);

    file->op = XDR_DECODE;
    file->xdr = NULL;

    format = sf_histstring(file,"data_format");
    if (NULL == format) {
	if (!sf_histint(file,"esize",&esize) || 0 != esize)
	    sf_error ("%s: Unknown format in %s",__FILE__,tag);
	sf_setformat(file,"ascii_float");
    } else {    
	sf_setformat(file,format);
	free (format);
    }

    return file;
}

/* Should do output after input */
sf_file sf_output (/*@null@*/ char* tag)
{
    sf_file file;
    char *headname, *dataname, *path, *name, *format;
    size_t namelen;

    file = (sf_file) sf_alloc(1,sizeof(*file));
    
    if (NULL == tag || 0 == strcmp(tag,"out")) {
	file->stream = stdout;
	headname = NULL;
    } else {
	headname = sf_getstring (tag);
	if (NULL == headname) {
	    namelen = strlen(tag)+1;
	    headname = sf_charalloc (namelen);
	    memcpy(headname,tag,namelen);
	}

	file->stream = fopen(headname,"w");
	if (NULL == file->stream) 
	    sf_error ("%s: Cannot write to header file %s:",__FILE__,headname);
    }
/*    setbuf(file->stream,file->buf); */
    
    file->pars = sf_simtab_init (tabsize);

    dataname = sf_getstring("out");
    if (NULL == dataname) {
	path = getdatapath();
	if (NULL == path) 
	    sf_error("%s: Cannot find datapath",__FILE__);
	file->dataname = sf_charalloc (PATH_MAX+NAME_MAX+1);
	strcpy (file->dataname,path);
	name = file->dataname+strlen(path);
	free (path);
	if (getfilename (file->stream,name)) {
	    namelen = strlen(file->dataname);
	    file->dataname[namelen]='@';
	    file->dataname[namelen+1]='\0';
	} else { /* invent a name */
	    sprintf(name,"%sXXXXXX",sf_getprog());
	    (void) close(mkstemp(file->dataname));
	    (void) unlink(file->dataname);
	    if (NULL == headname && 
		-1L == fseek(file->stream,0L,SEEK_CUR) &&
		ESPIPE == errno &&
		0 != mkfifo (file->dataname, S_IRUSR | S_IWUSR))
		sf_error ("%s: Cannot make a pipe %s:",
			  __FILE__,file->dataname);
	} 
	
    } else {
	namelen = strlen(dataname)+1;
	file->dataname = sf_charalloc (namelen);
	memcpy(file->dataname,dataname,namelen);
	free (dataname);
    }

    sf_putstring(file,"in",file->dataname);    

    file->op = XDR_ENCODE;
    file->xdr = NULL;

    if (NULL != infile && 
	NULL != (format = sf_histstring(infile,"data_format"))) {
	sf_setformat(file,format);
	free (format);
    } else {
	sf_setformat(file,"native_float");
    }

    if (NULL != headname) free(headname);
    return file;
}

sf_datatype sf_gettype (sf_file file)
{
    return file->type;
}

sf_dataform sf_getform (sf_file file)
{
    return file->form;
}

void sf_settype (sf_file file, sf_datatype type)
{
    file->type = type;
}

void sf_setformat (sf_file file, const char* format)
{
    if (NULL != strstr(format,"float")) {
	file->type = SF_FLOAT;
	sf_putint(file,"esize",(int) sizeof(float));
    } else if (NULL != strstr(format,"int")) {
	file->type = SF_INT;
	sf_putint(file,"esize",(int) sizeof(int));
    } else if (NULL != strstr(format,"complex")) {
	file->type = SF_COMPLEX;
	sf_putint(file,"esize",(int) sizeof(float complex));
    } else {
	file->type = SF_CHAR;
	sf_putint(file,"esize",(int) sizeof(char));
    }
    
    if (0 == strncmp(format,"ascii_",6)) {
	file->form = SF_ASCII;
	if (NULL != file->xdr) {
	    free (file->xdr);
	    free (file->buf);
	}
	file->xdr = NULL;
	sf_putint(file,"esize",0);
    } else if (0 == strncmp(format,"xdr_",4)) {
	file->form = SF_XDR;
	if (NULL == file->xdr) {
	    file->xdr = (XDR*) sf_alloc(1,sizeof(XDR));
	    file->buf = sf_charalloc(BUFSIZ);
	    xdrmem_create(file->xdr,file->buf,BUFSIZ,file->op);
	}
    } else {
	file->form = SF_NATIVE;
	if (NULL != file->xdr) {
	    free (file->xdr);
	    free (file->buf);
	}
	file->xdr = NULL;
    }
}
    
static bool getfilename (FILE* fp, char *filename)
{
    DIR* dir;
    struct stat buf;
    struct dirent *dirp;
    bool success;

    dir = opendir(".");
    if (NULL == dir) sf_error ("%s: cannot open current directory:",__FILE__);
    
    if(0 > fstat(fileno(fp),&buf)) sf_error ("%s: cannot run fstat:",__FILE__);
    success = false;
    
    while (NULL != (dirp = readdir(dir))) {
	if (dirp->d_ino == buf.st_ino) { /* non-portable */
	    strcpy(filename,dirp->d_name);
	    success = true;
	    break;
	}
    }

    return success;
}

static char* getdatapath (void) 
{
    char *path, *home, file[PATH_MAX];
    
    path = sf_getstring ("datapath");
    if (NULL != path) return path;

    path = getenv("DATAPATH");
    if (NULL != path) return path;

    path = sf_charalloc(PATH_MAX+1);
    if (readpathfile (".datapath",path)) return path;
    
    home = getenv("HOME");
    if (NULL != home) {
	snprintf(file,PATH_MAX,"%s/.datapath",home);
	if (readpathfile (file,path)) return path;
    }

    return NULL;
}

static bool readpathfile (const char* filename, char* datapath) 
{
    FILE *fp;
    char format[PATH_MAX];

    fp = fopen(filename,"r");
    if (NULL == fp) return false;

    if (0 >= fscanf(fp,"datapath=%s",datapath))
	sf_error ("No datapath found in file %s",filename);

    snprintf(format,PATH_MAX,"%s datapath=%%s",sf_gethost());
    (void) fscanf(fp,format,datapath);

    (void) fclose (fp);
    return true;
}

void sf_fileclose (sf_file file) 
{
    if (file->stream != stdin && 
	file->stream != stdout && 
	file->stream != NULL) (void) fclose (file->stream);
    if (NULL != file->pars) sf_simtab_close (file->pars);
    if (NULL != file->xdr) {
	free (file->xdr);
	free (file->buf);
    }
    free (file->dataname);
    free (file);
}

bool sf_histint (sf_file file, const char* key,/*@out@*/ int* par) 
{
    return sf_simtab_getint (file->pars,key,par);
}

bool sf_histfloat (sf_file file, const char* key,/*@out@*/ float* par) 
{
    return sf_simtab_getfloat (file->pars,key,par);
}

char* sf_histstring (sf_file file, const char* key) 
{
    return sf_simtab_getstring (file->pars,key);
}

void sf_fileflush (sf_file file, sf_file src)
{
    int i, n;
    float f;
    char key[8], *val;
    time_t tm;
    const char eol='\014', eot='\004';

    if (NULL == file->dataname) return;
    
    tm = time (NULL);
    if (0 >= fprintf(file->stream,"%s:\t%s@%s\t%s\n",
		     sf_getprog(),
		     sf_getuser(),
		     sf_gethost(),
		     ctime(&tm)))
	sf_error ("%s: cannot flush header:",__FILE__);
    switch (file->type) {
	case SF_FLOAT: 
	    switch (file->form) {
		case SF_ASCII:		    
		    sf_putstring(file,"data_format","ascii_float");
		    break;
		case SF_XDR:
		    sf_putstring(file,"data_format","xdr_float");
		    break;
		default:
		    sf_putstring(file,"data_format","native_float");
		    break;
	    }
	    break;
	case SF_COMPLEX:
	    switch (file->form) {
		case SF_ASCII:		    
		    sf_putstring(file,"data_format","ascii_complex");
		    break;
		case SF_XDR:
		    sf_putstring(file,"data_format","xdr_complex");
		    break;
		default:
		    sf_putstring(file,"data_format","native_complex");
		    break;
	    }
	    break;
	case SF_INT:
	    switch (file->form) {
		case SF_ASCII:		    
		    sf_putstring(file,"data_format","ascii_int");
		    break;
		case SF_XDR:
		    sf_putstring(file,"data_format","xdr_int");
		    break;
		default:
		    sf_putstring(file,"data_format","native_int");
		    break;
	    }
	    break;
	default:
	    sf_putstring(file,"data_format",
			 (NULL==file->xdr)? "native_byte":"xdr_byte");
	    break;
    }    
    if (NULL != src && NULL != src->pars) {	
	for (i=1; i <= SF_MAX_DIM; i++) {
	    snprintf(key,4,"n%d",i);
	    if (!sf_simtab_getint(src->pars,key,&n)) break;
	    if (0 >= fprintf(file->stream,"\t%s=%d\n",key,n))
		sf_error ("%s: cannot flush %s:",__FILE__,key);
	    snprintf(key,4,"o%d",i);
	    if (sf_simtab_getfloat(src->pars,key,&f) &&
		0 >= fprintf(file->stream,"\t%s=%g\n",key,f))
		sf_error ("%s: cannot flush %s:",__FILE__,key);
	    snprintf(key,4,"d%d",i);
	    if (sf_simtab_getfloat(src->pars,key,&f) &&
		0 >= fprintf(file->stream,"\t%s=%g\n",key,f))
		sf_error ("%s: cannot flush %s:",__FILE__,key);
	    snprintf(key,8,"label%d",i);
	    if (NULL != (val=sf_simtab_getstring(src->pars,key)) &&
		0 >= fprintf(file->stream,"\t%s=\"%s\"\n",key,val))
		sf_error ("%s: cannot flush %s:",__FILE__,key);
	}
    }
    
    sf_simtab_output(file->pars,file->stream);
    
    if (0==strcmp(file->dataname,"stdout")) { 
/* keep stream, write the header end code */
	fprintf(file->stream,"\tin=\"stdin\"\n\n%c%c%c",eol,eol,eot);
    } else {
	file->stream = freopen(file->dataname,"wb",file->stream);
	if (NULL == file->stream) 
	    sf_error ("%s: Cannot write to data file %s:",
		      __FILE__,file->dataname);
    }

    free (file->dataname);
    file->dataname=NULL;
}

void sf_putint (sf_file file, char* key, int par)
{
    char val[256];

    snprintf(val,256,"%d",par);
    sf_simtab_enter (file->pars,key,val);
}

void sf_putfloat (sf_file file, char* key,float par)
{
    char val[256];

    snprintf(val,256,"%g",par);
    sf_simtab_enter (file->pars,key,val);
}

void sf_putstring (sf_file file, char* key,const char* par)
{
    char *val;
    
    val = (char*) alloca(strlen(par)+3); 
    sprintf(val,"\"%s\"",par);
    sf_simtab_enter (file->pars,key,val);
}

void sf_putline (sf_file file, const char* line)
{
    if (0 >= fprintf(file->stream,"\t%s\n",line))
	sf_error ("%s: cannot put line '%s':",__FILE__,line);
}

void sf_setaformat (const char* format, int line)
{
    size_t len;

    if (NULL != aformat) free (aformat);
    if (NULL != format) {
	len = strlen(format)+1;
	aformat = sf_charalloc(len);
	memcpy(aformat,format,len);
    } else {
	aformat = NULL;
    }
    aline = (size_t) line;
}

void sf_write (void* arr, size_t esize, size_t size, sf_file file)
{
    char* buf;
    size_t i, left, nbuf;
    bool_t success;
    float complex c;

    if (NULL != file->dataname) sf_fileflush (file,infile);
    switch(file->form) {
	case SF_ASCII:
	    for (left = size; left > 0; left-=nbuf) {
		nbuf = (aline < left)? aline: left;
		switch(file->type) {
		    case SF_INT:
			for (i=size-left; i < size-left+nbuf; i++) {
			    if (EOF==fprintf(file->stream,
					     (NULL != aformat)? aformat:"%d ",
					     ((int*)arr)[i]))
				sf_error ("%s: trouble writing ascii",__FILE__);
			}
			break;
		    case SF_FLOAT:
			for (i=size-left; i < size-left+nbuf; i++) {
			    if (EOF==fprintf(file->stream,
					     (NULL != aformat)? aformat:"%g ",
					     ((float*)arr)[i]))
				sf_error ("%s: trouble writing ascii",__FILE__);
			}
			break;
		    case SF_COMPLEX:
			for (i=size-left; i < size-left+nbuf; i++) {
			    c = ((float complex*)arr)[i];
			    if (EOF==fprintf(file->stream,
					     (NULL != aformat)? 
					     aformat:"(%g,%g) ",
					     crealf(c),cimagf(c)))
				sf_error ("%s: trouble writing ascii",__FILE__);
			}
			break;
		    default:
			for (i=size-left; i < size-left+nbuf; i++) {
			    if (EOF==fputc(((char*)arr)[i],file->stream))
				sf_error ("%s: trouble writing ascii",__FILE__);
			}
			break;
		}
		if (EOF==fprintf(file->stream,"\n"))
		    sf_error ("%s: trouble writing ascii",__FILE__);
	    }
	    break;
	case SF_XDR:
	    size *= esize;
	    buf = (char*)arr+size;
	    for (left = size; left > 0; left -= nbuf) {
		nbuf = (BUFSIZ < left)? BUFSIZ : left;
		(void) xdr_setpos(file->xdr,0);
		switch(file->type) {
		    case SF_INT:
			success=xdr_vector(file->xdr,buf-left,
				     nbuf/sizeof(int),sizeof(int),
				     (xdrproc_t) xdr_int);
			break;
		    case SF_FLOAT:
		    case SF_COMPLEX:
			success=xdr_vector(file->xdr,buf-left,
				nbuf/sizeof(float),sizeof(float),
					(xdrproc_t) xdr_float);
			break;
		    default:
			success=xdr_opaque(file->xdr,buf-left,nbuf);
			break;
		}       
		if (0 == success) sf_error ("sf_file: trouble writing xdr");
		if (nbuf != fwrite(file->buf,1,nbuf,file->stream)) 
		    sf_error ("%s: trouble writing:",__FILE__);
	    }
	    break;
	default:
	    if (size != fwrite(arr,esize,size,file->stream)) 
		sf_error ("%s: trouble writing:",__FILE__);
	    break;
    }
}

void sf_read (/*@out@*/ void* arr, size_t esize, size_t size, sf_file file)
{
    char* buf;
    size_t i, left, nbuf, got;
    bool_t success;
    int c;
    float re, im;

    switch (file->form) {
	case SF_ASCII:
	    switch (file->type) {
		case SF_INT:
		    for (i = 0; i < size; i++) {
			if (EOF==fscanf(file->stream,"%d",(int*)arr+i))
			    sf_error ("%s: trouble reading ascii:",__FILE__);
		    }
		    break;
		case SF_FLOAT:
		    for (i = 0; i < size; i++) {
			if (EOF==fscanf(file->stream,"%g",(float*)arr+i))
			    sf_error ("%s: trouble reading ascii:",__FILE__);
		    }
		    break;
		case SF_COMPLEX:
		    for (i = 0; i < size; i++) {
			if (EOF==fscanf(file->stream,"(%g,%g)",&re,&im))
			    sf_error ("%s: trouble reading ascii:",__FILE__);
			((float complex*)arr)[i]=re+I*im;
		    }
		    break;
		default:
		    for (i = 0; i < size; i++) {
			c=fgetc(file->stream);
			if (EOF==c)
			    sf_error ("%s: trouble reading ascii:",__FILE__);
			((char*)arr)[i]= (char) c;
		    }
		    break;
	    }
	    break;
	case SF_XDR:
	    size *= esize;
	    buf = (char*)arr+size;
	    for (left = size; left > 0; left -= nbuf) {
		nbuf = (BUFSIZ < left)? BUFSIZ : left;
		(void) xdr_setpos(file->xdr,0);
		if (nbuf != fread(file->buf,1,nbuf,file->stream))
		    sf_error ("%s: trouble reading:",__FILE__);
		switch (file->type) {
		    case SF_INT:
			success=xdr_vector(file->xdr,buf-left,
				     nbuf/sizeof(int),sizeof(int),
				     (xdrproc_t) xdr_int);
			break;
		    case SF_FLOAT:
		    case SF_COMPLEX:
			success=xdr_vector(file->xdr,buf-left,
				     nbuf/sizeof(float),sizeof(float),
				     (xdrproc_t) xdr_float);
			break;
		    default:
			success=xdr_opaque(file->xdr,buf-left,nbuf);
			break;
		}
		if (0==success) sf_error ("%s: trouble reading xdr",__FILE__);
	    }
	    break;
	default:
	    got = fread(arr,esize,size,file->stream);
	    if (got != size) 
		sf_error ("%s: trouble reading: %d of %d",__FILE__,got,size);
	    break;
    }
}

long sf_bytes (sf_file file)
{
    int st;
    long size;
    struct stat buf;
    
    if (0 == strcmp(file->dataname,"stdin")) return -1L;

    if (NULL == file->dataname) {
	st = fstat(fileno(file->stream),&buf);
    } else {
	st = stat(file->dataname,&buf);
    }
    if (0 != st) sf_error ("%s: cannot find file size:",__FILE__);
    size = buf.st_size;
    return size;
}
    
long sf_tell (sf_file file)
{
    return ftell(file->stream);
}

void sf_seek (sf_file file, long offset, int whence)
{
    if (0 > fseek(file->stream,offset,whence))
	sf_error ("%s: seek problem:",__FILE__);
}

void sf_unpipe (sf_file file, size_t size) 
{
    size_t nbuf;
    char *path, *dataname;
    FILE* tmp;
    char buf[BUFSIZ];

    if (-1L != fseek(file->stream,0L,SEEK_CUR)) return;
    if (ESPIPE != errno) sf_error ("%s: pipe problem:",__FILE__);
	
    path = getdatapath();
    if (NULL == path) 
	sf_error ("%s: Cannot find datapath",__FILE__);
    dataname = sf_charalloc (NAME_MAX+1);
    snprintf(dataname,NAME_MAX,"%s%sXXXXXX",path,sf_getprog());
    tmp = fdopen(mkstemp(dataname),"wb");
    if (NULL == fdopen) sf_error ("%s: cannot open %s:",__FILE__,dataname);

    while (size > 0) {
	nbuf = (BUFSIZ < size)? BUFSIZ : size;
	if (nbuf != fread(buf,1,nbuf,file->stream) ||
	    nbuf != fwrite(buf,1,nbuf,tmp))
	    sf_error ("%s: trouble unpiping:",__FILE__);
	size -= nbuf;
    }
    (void) fclose(file->stream);
    file->stream = freopen(dataname,"rb",tmp);
    if (NULL == file->stream)
	sf_error ("%s: Trouble reading data file %s:",__FILE__,dataname);
} 
