# 🆓 Free AI Setup Guide - Google Gemini

Get real AI capabilities in your Smart Project Allocation System **completely free** using Google Gemini!

## 🚀 Quick Setup (5 minutes)

### Step 1: Get Your Free Gemini API Key
1. Go to **https://makersuite.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated API key (starts with `AIza...`)

### Step 2: Set Environment Variable

**On Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_key_here"
```

**On Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY=your_key_here
```

**On Mac/Linux:**
```bash
export GOOGLE_API_KEY=your_key_here
```

### Step 3: Install Required Package
```powershell
pip install --upgrade google-generativeai==0.8.3
```

### Step 4: Test AI Integration
```powershell
python test_claude_ai.py
```

You should see:
```
✅ GOOGLE_API_KEY found: AIza...
✅ Gemini AI client initialized successfully
🎉 AI integration test PASSED!
```

### Step 5: Run Your App
```powershell
python app.py
```

Now when you visit the project matching page, you'll see:
- **"Google Gemini AI"** green badge
- AI-generated match reasons and insights
- Intelligent project-employee compatibility analysis

## 🆓 Gemini Free Tier Limits

Google Gemini offers a **generous free tier**:
- ✅ **60 requests per minute**
- ✅ **1,500 requests per day**
- ✅ **100,000 tokens per day**
- ✅ **No credit card required**

This is perfect for development and demo purposes!

## 🔄 Fallback System

The system automatically handles different scenarios:

1. **Gemini Available** → Uses Google Gemini AI (Free!)
2. **Claude Available** → Uses Claude AI (Paid)
3. **No AI Keys** → Uses smart algorithm-based matching

## 🧪 Testing Different Scenarios

Test your setup:
```powershell
# Test with Gemini
$env:GOOGLE_API_KEY="your_gemini_key"
python test_claude_ai.py

# Test fallback (remove AI keys)
Remove-Item Env:GOOGLE_API_KEY
Remove-Item Env:ANTHROPIC_API_KEY
python test_claude_ai.py
```

## 🎯 Expected Results

With Gemini AI, you'll get intelligent insights like:

**✅ AI Reasons:**
- "Strong Python expertise (8/10) aligns perfectly with backend requirements"
- "5 years experience provides ideal technical leadership capability"
- "Current availability allows immediate project onboarding"

**⚠️ AI Concerns:**
- "May benefit from additional machine learning training"
- "Heavy current workload could impact initial velocity"

## 🚨 Troubleshooting

**API Key Issues:**
```
❌ Failed to initialize Gemini: 403 API_KEY_INVALID
```
→ Double-check your API key is correct and active

**Import Errors:**
```
❌ Import error: No module named 'google.generativeai'
```
→ Run: `pip install google-generativeai==0.3.2`

**Network Issues:**
```
❌ AI Analysis failed: Request timeout
```
→ Check internet connection and try again

## 🎉 Success!

Once set up, your project will have **real AI capabilities** that provide:
- 🧠 **Intelligent Analysis** - Contextual understanding of projects and skills
- 📊 **Smart Scoring** - AI-driven compatibility scores
- 💡 **Detailed Reasoning** - Explanations for every recommendation
- ⚠️ **Gap Analysis** - Identifies potential concerns and improvements

**All completely free with Google Gemini!** 🆓✨