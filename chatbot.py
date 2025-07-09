import pandas as pd
import streamlit as st
import json
import time
import random

# Custom CSS with all animations included
def inject_custom_css():
    st.markdown("""
    <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.03); }
            100% { transform: scale(1); }
        }
        
        .chat-message {
            animation: fadeIn 0.3s ease-out;
            margin: 10px 0;
            padding: 15px;
            border-radius: 15px;
            transition: all 0.2s;
        }
        
        .user-message {
            background-color: #f0f2f6;
            color: black;
            margin-left: 20%;
            border-bottom-right-radius: 5px !important;
        }
        
        .bot-message {
            background-color: #4f8bf9;
            color: white;
            margin-right: 20%;
            border-bottom-left-radius: 5px !important;
        }
        
        .stButton>button {
            animation: pulse 2s infinite;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
        }
        
        .stTextInput>div>div>input {
            padding: 12px !important;
            border-radius: 15px !important;
        }
        
        .header {
            background: linear-gradient(90deg, #4f8bf9, #6a5acd);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        
        .sidebar .stButton>button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Custom colored header replacement
def colored_header(label, description=None, color_name="blue-70"):
    st.markdown(f"""
    <div class="header">
        <h1 style="margin:0;padding:0">{label}</h1>
        <p style="margin:0;padding:0;font-size:0.9rem">{description or ''}</p>
    </div>
    """, unsafe_allow_html=True)

# Confetti effect using pure HTML/JS
def show_confetti():
    st.components.v1.html("""
    <canvas id="confetti" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:9999;pointer-events:none"></canvas>
    <script>
    const canvas = document.getElementById('confetti');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
    const confettiPieces = [];
    
    for (let i = 0; i < 150; i++) {
        confettiPieces.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height - canvas.height,
            r: Math.random() * 4 + 1,
            d: Math.random() * 3 + 2,
            color: colors[Math.floor(Math.random() * colors.length)]
        });
    }
    
    function drawConfetti() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        confettiPieces.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();
            
            p.y += p.d;
            if (p.y > canvas.height) {
                p.y = -10;
                p.x = Math.random() * canvas.width;
            }
        });
        
        requestAnimationFrame(drawConfetti);
    }
    
    setTimeout(() => {
        canvas.remove();
    }, 3000);
    
    drawConfetti();
    </script>
    """, height=0)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.first_interaction = True

# Main app
def main():
    inject_custom_css()
    
    # Header with custom colored header
    colored_header(
        label="🤖 BotBuddy - Your Shopping Assistant",
        description="Ask me anything about orders, returns, or products!",
        color_name="blue-70"
    )
    
    # Chat container
    chat_container = st.container()
    
    # Sidebar with info
    with st.sidebar:
        st.subheader("✨ Quick Actions")
        if st.button("🛍 Track My Order"):
            user_msg = "Track my order"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            bot_response = generate_response(user_msg)
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.rerun()

        if st.button("🔄 Start a Return"):
            user_msg = "Start a Return"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            bot_response = generate_response(user_msg)
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.rerun()
        if st.button("💬 Contact Support"):
            user_msg = "Contact Support"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            bot_response = generate_response(user_msg)
            st.session_state.messages.append({"role": "bot", "content": bot_response})
            st.rerun()
        st.divider()
        st.markdown("### 💡 Tips")
        st.info("• Try natural questions like 'Where's my package?'\n• Type 'help' for assistance\n• Say 'bye' when done")
    
    # Display chat messages
    with chat_container:
        if st.session_state.first_interaction:
            st.markdown("""
            <div style="text-align:center;margin:2rem 0">
                <h3>Welcome to BotBuddy! 👋</h3>
                <p>Your personal shopping assistant</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="chat-message bot-message" style="animation-delay: 0.5s;">
                <strong>BotBuddy:</strong> Hi there! I'm here to help with:
                <ul>
                    <li>Order tracking</li>
                    <li>Returns & exchanges</li>
                    <li>Product questions</li>
                    <li>Payment issues</li>
                </ul>
                Try asking: "Where's my order?" or "How do I return something?"
            </div>
            """, unsafe_allow_html=True)
            st.session_state.first_interaction = False
        
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message" style="animation-delay: {i*0.1}s;">
                    <strong></strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message" style="animation-delay: {i*0.1}s;">
                    <strong></strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # User input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", key="input", 
                                 placeholder="Ask me anything...", label_visibility="collapsed")
        
        cols = st.columns([5, 1])
        with cols[0]:
            if st.form_submit_button("Send", use_container_width=True):
                if user_input.strip():
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Show thinking animation
                    with chat_container:
                        with st.spinner("BotBuddy is thinking..."):
                            time.sleep(1)  # Simulate processing time
                            
                            # Generate response (replace with your FAQ logic)
                            bot_response = generate_response(user_input)
                            
                            # Add bot response
                            st.session_state.messages.append({"role": "bot", "content": bot_response})
                            
                            # Show confetti for thanks
                            if any(word in user_input.lower() for word in ["thank", "thanks", "appreciate"]):
                                show_confetti()
                    
                    # Rerun to update UI
                    st.rerun()
        
        with cols[1]:
            if st.form_submit_button("🎯", help="Get random suggestion", use_container_width=True):
                suggestions = [
                    "Where's my order?",
                    "How do I return something?",
                    "What's your return policy?",
                    "Do you offer student discounts?"
                ]
                st.session_state.messages.append({"role": "user", "content": random.choice(suggestions)})
                st.rerun()

# Response generator (replace with your actual FAQ logic)
def generate_response(query):
    query = query.lower()
    
    # Greetings
    if any(word in query for word in ["hi", "hello", "hey"]):
        return random.choice([
            "Hello there! 😊 How can I assist you today?",
            "Hi! Welcome back. What can I help you with?",
            "Hey friend! What would you like to know today?"
        ])
    
    # Common questions
    elif "track" in query:
        return "You can track your order using the link in your confirmation email or by visiting our website's order status page."
    
    elif "return" in query:
        return "Our return policy allows returns within 30 days with receipt. Start your return at our website's Returns Center."
    
    elif "contact" in query:
        return "You can reach our support team 24/7 at support@codealpha.com or via live chat on our website."
    
    # Farewells
    elif any(word in query for word in ["bye", "goodbye"]):
        return random.choice([
            "Goodbye! Come back if you have more questions! ❤️",
            "See you later! Don't hesitate to return if you need help.",
            "Have a wonderful day! 🌟"
        ])
    
    # Default response
    return "I'm still learning! For now, I can help with order tracking, returns, and general questions about our products."

if __name__ == "__main__":
    main()