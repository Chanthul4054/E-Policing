import os
import glob

html_files = glob.glob(r'd:\IIT\Y2S1\DSGP\E-Policing\templates\*.html')
html_files = [f for f in html_files if '403.html' not in f and 'login.html' not in f]

tab_html = """
                <li>
                    <a href="{{ url_for('records.index') }}" class="nav-item {% if request.endpoint == 'records.index' %}active{% endif %}">
                        <i class="icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                                <line x1="16" y1="13" x2="8" y2="13"></line>
                                <line x1="16" y1="17" x2="8" y2="17"></line>
                                <polyline points="10 9 9 9 8 9"></polyline>
                            </svg>
                        </i>
                        <span class="link-text">Past Crime Records</span>
                        <span class="tooltip">Past Crime Records</span>
                    </a>
                </li>"""

for file in html_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "Past Crime Records" not in content:
        parts = content.rsplit('</ul>', 1)
        if len(parts) == 2:
            new_content = parts[0] + tab_html + "\n            </ul>" + parts[1]
            with open(file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("Updated:", file)
