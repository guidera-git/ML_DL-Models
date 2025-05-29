import pandas as pd
import random

# Degree programs and their trait profiles
degree_program_traits = {
    "MBBS": [3, 3, 4, 2, 5, 5, 3, 2],
    "BDS": [3, 3, 3, 3, 5, 4, 3, 2],
    "DPT": [3, 3, 4, 2, 4, 5, 3, 2],
    "Pharm-D": [4, 3, 3, 2, 5, 4, 1, 2],
    "BS Nursing": [3, 3, 4, 2, 4, 5, 3, 2],
    "BS Electrical Engineering": [4, 5, 3, 3, 4, 2, 2, 3],
    "BS Mechanical Engineering": [4, 4, 2, 4, 4, 2, 2, 3],
    "BS Civil Engineering": [4, 4, 3, 3, 4, 2, 2, 3],
    "BS Software Engineering": [4, 5, 3, 4, 3, 2, 1, 1],
    "BS Computer Science": [5, 5, 3, 3, 3, 2, 1, 1],
    "BS Chemical Engineering": [4, 4, 3, 3, 5, 2, 2, 3],
    "BS Aerospace Engineering": [5, 5, 3, 4, 4, 1, 2, 3],
    "BS Biomedical Engineering": [4, 3, 4, 3, 5, 3, 3, 2],
    "BS Data Science": [5, 4, 3, 3, 4, 2, 1, 1],
    "BS Cyber Security": [5, 5, 2, 3, 5, 1, 1, 1],
    "BS Artificial Intelligence": [5, 4, 3, 4, 3, 2, 1, 1]
}

# Grade → % mapping for Cambridge
grade_mapping = {"A*": 95, "A": 85, "B": 75, "C": 65, "D": 55, "E": 45}

# University structure
university_structure = {
    "University of the Punjab (PU)": [
        "BS Software Engineering", "BS Computer Science", "BS Electrical Engineering",
        "BS Data Science", "BS Information Technology"
    ],
    "Lahore University of Management Sciences (LUMS)": [
        "BS Computer Science", "BS Chemical Engineering", "BS Electrical Engineering"
    ],
    "University of Central Punjab (UCP)": [
        "BS Software Engineering", "Pharm-D", "BS Cyber Security", "BS Computer Science",
        "BS Mechanical Engineering", "BS Civil Engineering", "BS Electrical Engineering",
        "BS Artificial Intelligence", "BS Biomedical Engineering"
    ],
    "The University of Lahore (UOL)": [
        "BS Civil Engineering", "Pharm-D", "BS Chemical Engineering",
        "BS Mechanical Engineering", "MBBS", "BS Electrical Engineering",
        "BS Software Engineering", "BS Computer Science", "BS Artificial Intelligence", "BDS"
    ],
    "University of Engineering and Technology (UET)": [
        "BS Mechanical Engineering", "BS Civil Engineering", "BS Electrical Engineering",
        "BS Software Engineering", "BS Computer Science", "BS Artificial Intelligence",
        "BS Biomedical Engineering", "BS Data Science"
    ],
    "University of Management and Technology (UMT)": [
        "BS Civil Engineering", "Pharm-D", "BS Chemical Engineering", "BS Mechanical Engineering",
        "MBBS", "BS Electrical Engineering", "BS Software Engineering",
        "BS Computer Science", "BS Biomedical Engineering", "BS Artificial Intelligence", "BDS"
    ],
    "Beaconhouse National University (BNU)": [
        "BS Computer Science", "BS Software Engineering", "BS Artificial Intelligence"
    ],
    "Forman Christian College (FCCU)": [
        "BS Computer Science", "BS Biomedical Engineering"
    ],
    "National University of Sciences and Technology (NUST)": [
        "BS Computer Science", "BS Aerospace Engineering", "BS Chemical Engineering",
        "BS Artificial Intelligence", "BS Mechanical Engineering", "BS Electrical Engineering"
    ],
    "COMSATS University Islamabad": [
        "BS Civil Engineering", "BS Chemical Engineering", "BS Mechanical Engineering",
        "BS Electrical Engineering", "BS Software Engineering", "BS Computer Science",
        "BS Biomedical Engineering", "BS Artificial Intelligence"
    ],
    "FAST-NU": [
        "BS Computer Science", "BS Software Engineering", "BS Data Science",
        "BS Civil Engineering", "BS Artificial Intelligence", "BS Electrical Engineering",
        "BS Cyber Security"
    ],
    "The University of Education (UE)": [
        "BS Computer Science", "BS Electrical Engineering", "BS Software Engineering",
        "Pharm-D", "DPT"
    ],
    "Pakistan Institute of Engineering and Applied Sciences (PIEAS)": [
        "BS Software Engineering", "BS Chemical Engineering", "BS Computer Science",
        "BS Mechanical Engineering", "BS Artificial Intelligence", "BS Data Science",
        "BS Cyber Security", "BS Civil Engineering"
    ],
    "Quaid-i-Azam University (QAU)": ["BS Computer Science"],
    "Riphah International University": [
        "BS Software Engineering", "BS Computer Science", "BS Data Science",
        "BS Civil Engineering", "BS Artificial Intelligence", "BS Electrical Engineering",
        "BS Cyber Security", "DPT", "BS Nursing", "MBBS", "BDS", "Pharm-D"
    ],
    "Aga Khan University (AKU)": ["MBBS"],
    "King Edward Medical University (KEMU)": [
        "MBBS", "DPT", "BS Nursing", "BS ALLIED VISION SCIENCES"
    ],
    "Lahore College for Women University (LCWU)": [
        "BS Computer Science", "BS Information Technology"
    ],
    "Ghulam Ishaq Khan Institute (GIKI)": [
        "BS Software Engineering", "BS Chemical Engineering", "BS Computer Science",
        "BS Mechanical Engineering", "BS Artificial Intelligence", "BS Data Science",
        "BS Cyber Security", "BS Civil Engineering"
    ],
    "Information Technology University (ITU)": [
        "BS Computer Science", "BS Data Science", "BS Software Engineering",
        "BS Artificial Intelligence", "BS Electrical Engineering"
    ]
}

# Which universities are “big”
big_universities = {
    "Ghulam Ishaq Khan Institute (GIKI)",
    "Aga Khan University (AKU)",
    "King Edward Medical University (KEMU)",
    "FAST-NU",
    "National University of Sciences and Technology (NUST)",
    "COMSATS University Islamabad",
    "University of the Punjab (PU)",
    "University of Engineering and Technology (UET)",
    "Lahore University of Management Sciences (LUMS)"
}

# Streams
study_streams = ["Pre-Medical", "Pre-Engineering", "Computer Science"]

# Simulation parameters
N = 100
noise_ratio = 0.2  # 20% noisy records

data = []

for _ in range(N):
    is_noisy = (random.random() < noise_ratio)
    
    # Gender (1=male, 0=female)
    gender = random.choice(["1", "0"])
    
    # Academic percentage
    is_cambridge = (random.random() < 0.10)
    if is_cambridge:
        o_lvl = random.choice(list(grade_mapping))
        a_lvl = random.choice(list(grade_mapping))
        pct = round((grade_mapping[o_lvl] + grade_mapping[a_lvl]) / 2, 2)
    else:
        m_marks = random.randint(633, 1100)
        i_marks = random.randint(633, 1100)
        pct = round(((m_marks + i_marks) / 2200) * 100, 2)
    
    # Study stream
    stream = random.choice(study_streams)
    
    # Determine allowed degrees based on stream
    allowed_degrees = list(degree_program_traits.keys())
    if stream in ("Pre-Engineering", "Computer Science"):
        for med in ["MBBS", "BDS", "DPT", "Pharm-D", "BS Nursing"]:
            if med in allowed_degrees:
                allowed_degrees.remove(med)
    
    # Ensure BS Nursing is only for females
    if gender == "1" and "BS Nursing" in allowed_degrees:
        allowed_degrees.remove("BS Nursing")
    
    # Degree & traits
    if is_noisy:
        degree = random.choice(allowed_degrees)
        traits = [random.randint(1, 5) for _ in range(6)] + [random.randint(1, 3) for _ in range(2)]

    else:
        degree = random.choice(allowed_degrees)
        traits = degree_program_traits[degree]
    
    # University assignment
    possibles = [u for u, progs in university_structure.items() if degree in progs]
    if pct < 75:
        possibles = [u for u in possibles if u not in big_universities]
    uni = random.choice(possibles) if possibles else "Unknown"
    
    data.append({
        "Gender": gender,
        "Academic Percentage": pct,
        "Study Stream": stream,
        "Degree Program": degree,
        "University": uni,
        "Analytical": traits[0],
        "Logical": traits[1],
        "Explaining": traits[2],
        "Creative": traits[3],
        "Detail-Oriented": traits[4],
        "Helping": traits[5],
        "Activity Preference": traits[6],
        "Project Preference": traits[7]
    })

# Save to Excel
df = pd.DataFrame(data)
df.to_excel("pakistani_student_survey_data_with_noise_filtered.xlsx", index=False)
print("✅ Generated: pakistani_student_survey_data_with_noise_filtered.xlsx")