def parse_xml_with_recovery(file_path):
    texts = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        xml_string = ''
        for line in lines:
            try:

                ET.fromstring('<root>' + line.strip() + '</root>')
                xml_string += line
            except ET.ParseError:

                continue
        
        root = ET.fromstring('<root>' + xml_string + '</root>')
        texts = [text_elem.text for text_elem in root.findall('.//TEXT')]
    except Exception as e:
        print(f"Errore generale: {e}")
    
    return texts

def filter_documents(docss):
    filtered_docss = []
    for doc_list in docss:
        filtered_list = [doc for doc in doc_list if len(doc.split()) >= 2]
        if filtered_list:
            filtered_docss.append(filtered_list)
    return filtered_docss
