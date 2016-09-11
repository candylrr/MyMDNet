import imagehash
def getimagehash(pic):
	img_hash=str(imagehash.dhash(Image.open(pic)))
	print img_hash
if __name__ == '__main__':
	getimagehash(str(sys.argv[1]))
