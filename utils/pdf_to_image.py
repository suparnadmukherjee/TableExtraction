# import module
import  os
from pdf2image import convert_from_path

# source_file_path="/home/suparna/PycharmProjects/TableDetection/all_Data/10k_reports/pdf/TransDigm.pdf"

# Store Pdf with convert_from_path function
def pdftoimage(pdf_path,png_pages_dir):
	'''

	Args:
		pdf_path: .pdf file
		png_pages_dir: dir path where the page images will be saved

	Returns:None

	'''
	filename = pdf_path.split('/')[-1][:-4]
	#path = os.path.join(png_pages_dir, filename)
	#os.mkdir(f"{path}/")
	images = convert_from_path(pdf_path)

	for i in range(len(images)):
		# Save pages as images in the pdf
		images[i].save(f'{png_pages_dir}{filename}_{str(i)}.png', 'PNG')

if __name__=="__main__":
	pdftoimage("/home/suparna/PycharmProjects/TableExtraction/data/pdf/Amazon.pdf",
			   "/home/suparna/PycharmProjects/TableExtraction/data/pngpages/")