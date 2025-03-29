import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import telebot
from telebot import types
import openai
from io import BytesIO
import plotly.express as px
import plotly.io as pio
from dotenv import load_dotenv
from flask import Flask, request
import time
import traceback

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)

# Initialize OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Initialize Telegram bot
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN environment variable not set!")
    # Use a fallback token for testing only
    TELEGRAM_TOKEN = "test"

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Load the data - check both locations
def load_data():
    try:
        # Try data.csv in root directory first
        try:
            data = pd.read_csv('data.csv')
            logger.info(f"Data loaded from root directory: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            logger.warning(f"Error loading data from root: {e}")
            
        # Try data directory
        try:
            data = pd.read_csv('data/data.csv')
            logger.info(f"Data loaded from data directory: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            logger.warning(f"Error loading data from data directory: {e}")
            
        # Try working directory
        import glob
        logger.info(f"Files in current directory: {glob.glob('*')}")
        logger.info(f"Current working directory: {os.getcwd()}")
            
        raise Exception("Could not find data.csv in any location")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# Function to generate insights using OpenAI in Azerbaijani
def generate_insights(data_description):
    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sən dəqiq və analitik neft və qaz emalı prosesləri üzrə məlumatlar haqqında Azərbaycan dilində təhlil təqdim edən köməkçisən."},
                {"role": "user", "content": f"Aşağıdakı məlumatları təhlil et və biznes üçün əhəmiyyətli nəticələri Azərbaycan dilində təqdim et: {data_description}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating insights with OpenAI: {e}")
        return "OpenAI ilə təhlil yaradılarkən xəta baş verdi."

# Function to create charts
def create_efficiency_chart(data):
    try:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Proses Tipi', y='Emalın Səmərəliliyi (%)', data=data)
        plt.title('Proses Tipinə görə Emal Səmərəliliyi', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        return buffer
    except Exception as e:
        logger.error(f"Error creating efficiency chart: {str(e)}")
        raise

def create_energy_chart(data):
    try:
        plt.figure(figsize=(12, 8))
        fig = px.scatter(data, 
                        x='Emal Həcmi (ton)', 
                        y='Enerji İstifadəsi (kWh)',
                        color='Proses Tipi',
                        size='Energy_per_ton',
                        hover_data=['Proses ID', 'Təzyiq (bar)', 'Temperatur (°C)'])
        fig.update_layout(title='Emal Həcmi və Enerji İstifadəsi Arasında Əlaqə')
        
        buffer = BytesIO()
        pio.write_image(fig, buffer, format='png')
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Error creating energy chart: {str(e)}")
        raise

def create_environmental_chart(data):
    try:
        plt.figure(figsize=(12, 8))
        avg_co2_by_type = data.groupby('Proses Tipi')['Ətraf Mühitə Təsir (g CO2 ekvivalent)'].mean().reset_index()
        avg_co2_by_type = avg_co2_by_type.sort_values('Ətraf Mühitə Təsir (g CO2 ekvivalent)', ascending=False)
        
        plt.barh(avg_co2_by_type['Proses Tipi'], avg_co2_by_type['Ətraf Mühitə Təsir (g CO2 ekvivalent)'])
        plt.title('Proses Tipinə görə Ortalama CO2 Emissiyası', fontsize=16)
        plt.xlabel('CO2 Emissiyası (g)', fontsize=12)
        plt.ylabel('Proses Tipi', fontsize=12)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        return buffer
    except Exception as e:
        logger.error(f"Error creating environmental chart: {str(e)}")
        raise

def create_cost_chart(data):
    try:
        plt.figure(figsize=(12, 8))
        cost_by_type = data.groupby('Proses Tipi')['Əməliyyat Xərcləri (AZN)'].sum().reset_index()
        cost_by_type = cost_by_type.sort_values('Əməliyyat Xərcləri (AZN)', ascending=False)
        
        plt.bar(cost_by_type['Proses Tipi'], cost_by_type['Əməliyyat Xərcləri (AZN)'] / 1000)
        plt.title('Proses Tipinə görə Ümumi Əməliyyat Xərcləri', fontsize=16)
        plt.xlabel('Proses Tipi', fontsize=12)
        plt.ylabel('Əməliyyat Xərcləri (Min AZN)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        return buffer
    except Exception as e:
        logger.error(f"Error creating cost chart: {str(e)}")
        raise

# Generate a summary of the data
def generate_data_summary(data):
    try:
        summary = {
            'total_processes': data.shape[0],
            'process_types': data['Proses Tipi'].nunique(),
            'avg_efficiency': data['Emalın Səmərəliliyi (%)'].mean(),
            'total_energy': data['Enerji İstifadəsi (kWh)'].sum(),
            'total_cost': data['Əməliyyat Xərcləri (AZN)'].sum(),
            'avg_co2': data['Ətraf Mühitə Təsir (g CO2 ekvivalent)'].mean(),
            'max_volume': data['Emal Həcmi (ton)'].max(),
            'safety_incidents': data['Təhlükəsizlik Hadisələri'].sum()
        }
        
        # Top and bottom performers
        top_efficiency = data.sort_values('Emalın Səmərəliliyi (%)', ascending=False).head(3)
        low_efficiency = data.sort_values('Emalın Səmərəliliyi (%)', ascending=True).head(3)
        
        # Most efficient process type
        process_efficiency = data.groupby('Proses Tipi')['Emalın Səmərəliliyi (%)'].mean().reset_index()
        best_process = process_efficiency.loc[process_efficiency['Emalın Səmərəliliyi (%)'].idxmax()]
        
        summary_text = f"""
Ümumi Məlumat Təhlili:
- Ümumi proses sayı: {summary['total_processes']}
- Fərqli proses tipləri: {summary['process_types']}
- Ortalama emal səmərəliliyi: {summary['avg_efficiency']:.2f}%
- Ümumi enerji istifadəsi: {summary['total_energy']:,} kWh
- Ümumi əməliyyat xərcləri: {summary['total_cost']:,} AZN
- Ortalama CO2 emissiyası: {summary['avg_co2']:,.2f} g
- Maksimum emal həcmi: {summary['max_volume']:,} ton
- Qeydə alınmış təhlükəsizlik hadisələri: {summary['safety_incidents']}

Ən Yüksək Səmərəliliyə Malik Proseslər:
"""
        
        for i, row in top_efficiency.iterrows():
            summary_text += f"- Proses ID: {row['Proses ID']}, Tipi: {row['Proses Tipi']}, Səmərəlilik: {row['Emalın Səmərəliliyi (%)']}%\n"
        
        summary_text += f"\nƏn Aşağı Səmərəliliyə Malik Proseslər:\n"
        
        for i, row in low_efficiency.iterrows():
            summary_text += f"- Proses ID: {row['Proses ID']}, Tipi: {row['Proses Tipi']}, Səmərəlilik: {row['Emalın Səmərəliliyi (%)']}%\n"
        
        summary_text += f"\nƏn Səmərəli Proses Tipi: {best_process['Proses Tipi']} (Ortalama {best_process['Emalın Səmərəliliyi (%)']}%)"
        
        return summary_text
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        raise

# Webhook handler (this is what actually works on Render.com)
@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    logger.info("Received webhook request")
    try:
        if request.headers.get('content-type') == 'application/json':
            json_str = request.get_data().decode('UTF-8')
            logger.info(f"Webhook data: {json_str[:100]}...")
            update = types.Update.de_json(json_str)
            logger.info(f"Processing update: {update.update_id}")
            
            # Extract chat_id and message for direct handling
            if hasattr(update, 'message') and update.message:
                chat_id = update.message.chat.id
                message_text = update.message.text if hasattr(update.message, 'text') else None
                logger.info(f"Detected chat_id: {chat_id}, message: {message_text}")
                
                # Handle new chat members (user joined)
                if hasattr(update.message, 'new_chat_members') and update.message.new_chat_members:
                    logger.info("New chat member detected, sending welcome message")
                    send_welcome_message(chat_id)
                    return ''
                
                # Handle commands manually
                if message_text == '/start':
                    logger.info("Detected /start command, handling directly")
                    send_welcome_message(chat_id)
                    logger.info("Welcome message sent with keyboard")
                
                elif message_text == '/help':
                    logger.info("Detected /help command")
                    help_text = """
SOCAR Process Analyst Bot - Kömək

Bu bot SOCAR neft və qaz emalı prosesləri üzrə məlumatların təhlili və vizualizasiyası üçün yaradılmışdır.

Mövcud əmrlər:
/start - Botu başlatmaq və əsas menyunu göstərmək
/help - Bu kömək mesajını göstərmək
/summary - Əsas məlumatların xülasəsini göstərmək
/menu - Əsas menyunu yenidən göstərmək

Panel düymələri vasitəsilə aşağıdakı təhlilləri əldə edə bilərsiniz:
- Əsas Məlumatlar: Proseslərin ümumi statistikası
- Səmərəlilik Analizi: Proses tipinə görə səmərəlilik göstəriciləri
- Enerji İstifadəsi: Enerji istifadəsi və emal həcmi arasında əlaqə
- Ətraf Mühit Təsiri: CO2 emissiyalarının təhlili
- Xərc Analizi: Əməliyyat xərclərinin təhlili
- OpenAI Təhlili: Süni intellekt tərəfindən yaradılmış təhlil

Əlavə məlumat üçün: ismetsemedov@gmail.com
"""
                    bot.send_message(chat_id, help_text)
                    logger.info("Help information sent to user")
                
                elif message_text == '/menu' or message_text == '/keyboard':
                    logger.info("Detected /menu command")
                    show_main_menu(chat_id)
                    logger.info("Main menu sent to user")
                
                elif message_text == '/summary':
                    logger.info("Detected /summary command")
                    bot.send_message(chat_id, "Əsas məlumatlar yüklənir...")
                    
                    try:
                        data = load_data()
                        if data is None:
                            bot.send_message(chat_id, "Məlumatların yüklənməsində xəta baş verdi.")
                            return ''
                        
                        summary = generate_data_summary(data)
                        bot.send_message(chat_id, summary)
                        logger.info("Summary sent to user")
                    except Exception as e:
                        logger.error(f"Error processing data: {e}")
                        logger.error(traceback.format_exc())
                        bot.send_message(chat_id, f"Məlumatların emalında xəta: {str(e)}")
                
                elif message_text == 'Əsas Məlumatlar':
                    logger.info("Handling 'Əsas Məlumatlar' request")
                    bot.send_message(chat_id, "Əsas məlumatlar yüklənir...")
                    
                    try:
                        # Load data
                        data = load_data()
                        if data is None:
                            bot.send_message(chat_id, "Məlumatların yüklənməsində xəta baş verdi.")
                            return ''
                            
                        logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
                        
                        # Generate summary
                        summary = generate_data_summary(data)
                        logger.info("Data summary generated")
                        
                        # Send summary
                        bot.send_message(chat_id, summary)
                        logger.info("Summary sent to user")
                    except Exception as e:
                        logger.error(f"Error processing data: {e}")
                        logger.error(traceback.format_exc())
                        bot.send_message(chat_id, f"Məlumatların emalında xəta: {str(e)}")
                
                elif message_text == 'Səmərəlilik Analizi':
                    logger.info("Handling 'Səmərəlilik Analizi' request")
                    bot.send_message(chat_id, "Səmərəlilik analizi hazırlanır...")
                    
                    try:
                        # Load data
                        data = load_data()
                        if data is None:
                            bot.send_message(chat_id, "Məlumatların yüklənməsində xəta baş verdi.")
                            return ''
                            
                        logger.info("Data loaded for efficiency analysis")
                        
                        # Create chart
                        chart = create_efficiency_chart(data)
                        logger.info("Efficiency chart created")
                        
                        # Send chart
                        bot.send_photo(chat_id, chart, caption="Proses Tipinə görə Emal Səmərəliliyi")
                        logger.info("Efficiency chart sent to user")
                    except Exception as e:
                        logger.error(f"Error creating chart: {e}")
                        logger.error(traceback.format_exc())
                        bot.send_message(chat_id, f"Qrafik yaradılarkən xəta: {str(e)}")
                
                elif message_text == 'Enerji İstifadəsi':
                    logger.info("Handling 'Enerji İstifadəsi' request")
                    bot.send_message(chat_id, "Enerji istifadəsi analizi hazırlanır...")
                    
                    try:
                        data = load_data()
                        if data is None:
                            bot.send_message(chat_id, "Məlumatların yüklənməsində xəta baş verdi.")
                            return ''
                            
                        chart = create_energy_chart(data)
                        bot.send_photo(chat_id, chart, caption="Emal Həcmi və Enerji İstifadəsi Arasında Əlaqə")
                        logger.info("Energy chart sent to user")
                    except Exception as e:
                        logger.error(f"Error with energy chart: {e}")
                        logger.error(traceback.format_exc())
                        bot.send_message(chat_id, f"Qrafik yaradılarkən xəta: {str(e)}")
                
                elif message_text == 'Ətraf Mühit Təsiri':
                    logger.info("Handling 'Ətraf Mühit Təsiri' request")
                    bot.send_message(chat_id, "Ətraf mühit təsiri analizi hazırlanır...")
                    
                    try:
                        data = load_data()
                        if data is None:
                            bot.send_message(chat_id, "Məlumatların yüklənməsində xəta baş verdi.")
                            return ''
                            
                        chart = create_environmental_chart(data)
                        bot.send_photo(chat_id, chart, caption="Proses Tipinə görə Ortalama CO2 Emissiyası")
                        logger.info("Environmental chart sent to user")
                    except Exception as e:
                        logger.error(f"Error with environmental chart: {e}")
                        logger.error(traceback.format_exc())
                        bot.send_message(chat_id, f"Qrafik yaradılarkən xəta: {str(e)}")
                
                elif message_text == 'Xərc Analizi':
                    logger.info("Handling 'Xərc Analizi' request")
                    bot.send_message(chat_id, "Xərc analizi hazırlanır...")
                    
                    try:
                        data = load_data()
                        if data is None:
                            bot.send_message(chat_id, "Məlumatların yüklənməsində xəta baş verdi.")
                            return ''
                            
                        chart = create_cost_chart(data)
                        bot.send_photo(chat_id, chart, caption="Proses Tipinə görə Ümumi Əməliyyat Xərcləri")
                        logger.info("Cost chart sent to user")
                    except Exception as e:
                        logger.error(f"Error with cost chart: {e}")
                        logger.error(traceback.format_exc())
                        bot.send_message(chat_id, f"Qrafik yaradılarkən xəta: {str(e)}")
                
                elif message_text == 'OpenAI Təhlili':
                    logger.info("Handling 'OpenAI Təhlili' request")
                    bot.send_message(chat_id, "OpenAI təhlili hazırlanır, xahiş edirik gözləyin...")
                    
                    try:
                        data = load_data()
                        if data is None:
                            bot.send_message(chat_id, "Məlumatların yüklənməsində xəta baş verdi.")
                            return ''
                            
                        summary = generate_data_summary(data)
                        insights = generate_insights(summary)
                        
                        if len(insights) > 4000:
                            chunks = [insights[i:i+4000] for i in range(0, len(insights), 4000)]
                            for chunk in chunks:
                                bot.send_message(chat_id, chunk)
                        else:
                            bot.send_message(chat_id, insights)
                        logger.info("OpenAI insights sent to user")
                    except Exception as e:
                        logger.error(f"Error with OpenAI analysis: {e}")
                        logger.error(traceback.format_exc())
                        bot.send_message(chat_id, f"Təhlil yaradılarkən xəta: {str(e)}")
                
                else:
                    # Default response for unknown commands
                    bot.send_message(chat_id, f"'{message_text}' əmri tanınmadı. Kömək üçün /help yazın və ya panel düymələrindən istifadə edin.")
                    logger.info(f"Sent default response for: {message_text}")
            
            return ''
        else:
            logger.warning(f"Received non-JSON content type: {request.headers.get('content-type')}")
            return '', 403
    except Exception as e:
        logger.error(f"Error in webhook processing: {e}")
        logger.error(traceback.format_exc())
        return '', 500

# Helper functions for the bot
def send_welcome_message(chat_id):
    """Send welcome message with bot information and show the main menu"""
    welcome_text = """
🔍 *SOCAR Process Analyst Bot*-a xoş gəlmisiniz!

Bu bot SOCAR neft və qaz emalı prosesləri üzrə məlumatların təhlili və vizualizasiyası üçün yaradılmışdır.

✅ *Nə edə bilər?*
• Proses səmərəliliyini analiz etmək
• Enerji istifadəsini vizuallaşdırmaq
• Ətraf mühitə təsiri ölçmək
• Əməliyyat xərclərini təhlil etmək
• AI təhlili ilə əlavə insights təqdim etmək

Daha ətraflı məlumat üçün /help yazın.
"""
    
    bot.send_message(chat_id, welcome_text, parse_mode='Markdown')
    show_main_menu(chat_id)

def show_main_menu(chat_id):
    """Display the main menu keyboard"""
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    item1 = types.KeyboardButton('Əsas Məlumatlar')
    item2 = types.KeyboardButton('Səmərəlilik Analizi')
    item3 = types.KeyboardButton('Enerji İstifadəsi')
    item4 = types.KeyboardButton('Ətraf Mühit Təsiri')
    item5 = types.KeyboardButton('Xərc Analizi')
    item6 = types.KeyboardButton('OpenAI Təhlili')
    
    markup.add(item1, item2, item3, item4, item5, item6)
    bot.send_message(chat_id, "Lütfən, analiz növünü seçin:", reply_markup=markup)


# Health check endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return 'Bot is running!'

@app.route('/test-data', methods=['GET'])
def test_data():
    try:
        data = load_data()
        if data is not None:
            return f"CSV loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
        else:
            return "Failed to load data.csv"
    except Exception as e:
        return f"Error loading data: {str(e)}"

@app.route('/')
def index():
    return 'Telegram Bot is running!'

if __name__ == '__main__':
    # Get the port from environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    
    # For production, use webhook mode
    if os.environ.get('ENVIRONMENT') == 'production':
        # Remove any existing webhook first
        bot.remove_webhook()
        time.sleep(0.5)  # Give Telegram servers some time to process
        
        # Set webhook
        url = os.environ.get('APP_URL', '')
        if url:
            webhook_url = f"{url}/{TELEGRAM_TOKEN}"
            bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook set to {webhook_url}")
        else:
            logger.error("APP_URL environment variable not set")
        
        # Only run the Flask app (no polling) in production
        app.run(host='0.0.0.0', port=port)
    else:
        # For development, just use polling mode
        bot.remove_webhook()
        logger.info("Starting bot in polling mode for development")
        bot.infinity_polling()