import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- 1. Generate a Mock Dataset ---
def generate_mock_data(num_records=10000):
    """Generates a mock dataset of Ubisoft player data."""
    np.random.seed(42) # for reproducibility

    countries = ['USA', 'Canada', 'Germany', 'UK', 'France', 'Brazil', 'Russia', 'China', 'Japan', 'Australia']
    genders = ['Male', 'Female', 'Non-Binary', 'Prefer not to say']
    # Common Ubisoft franchises/games (simplified)
    games = [
        'Assassin\'s Creed Valhalla', 'Far Cry 6', 'Rainbow Six Siege',
        'Watch Dogs: Legion', 'The Division 2', 'Ghost Recon Breakpoint',
        'Immortals Fenyx Rising', 'Riders Republic', 'Skull and Bones' # (hypothetical future play)
    ]
    genres = {
        'Assassin\'s Creed Valhalla': 'Action RPG',
        'Far Cry 6': 'FPS',
        'Rainbow Six Siege': 'Tactical FPS',
        'Watch Dogs: Legion': 'Action-Adventure',
        'The Division 2': 'Action RPG',
        'Ghost Recon Breakpoint': 'Tactical Shooter',
        'Immortals Fenyx Rising': 'Action-Adventure',
        'Riders Republic': 'Sports',
        'Skull and Bones': 'Action-Adventure'
    }

    data = {
        'PlayerID': range(1, num_records + 1),
        'Age': np.random.randint(13, 65, num_records),
        'Gender': np.random.choice(genders, num_records, p=[0.65, 0.30, 0.03, 0.02]),
        'Country': np.random.choice(countries, num_records, p=[0.25, 0.10, 0.12, 0.10, 0.08, 0.08, 0.07, 0.08, 0.06, 0.06]),
        'GamePlayed': np.random.choice(games, num_records),
    }

    # Assign hours played - some games inherently get more hours
    hours = []
    for game in data['GamePlayed']:
        if game in ['Assassin\'s Creed Valhalla', 'The Division 2', 'Rainbow Six Siege']: # RPGs/Live service
            hours.append(np.random.randint(50, 500))
        elif game in ['Far Cry 6', 'Watch Dogs: Legion']:
            hours.append(np.random.randint(20, 200))
        else:
            hours.append(np.random.randint(10, 150))
    data['HoursPlayed'] = hours
    data['PreferredGenre'] = [genres[game] for game in data['GamePlayed']] # Simplified: genre of game played

    df = pd.DataFrame(data)
    return df

# --- 2. Analysis Functions ---

def analyze_demographics(df):
    """Analyzes and visualizes player demographics (age, gender)."""
    print("\n--- Player Demographics ---")

    # Age Analysis
    print(f"Average Player Age: {df['Age'].mean():.2f} years")
    print(f"Median Player Age: {df['Age'].median()} years")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Player Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Number of Players')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Gender Analysis
    gender_counts = df['Gender'].value_counts(normalize=True) * 100
    print("\nGender Distribution:")
    print(gender_counts)
    plt.figure(figsize=(8, 8))
    gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
    plt.title('Player Gender Distribution')
    plt.ylabel('') # Hide the default 'Gender' ylabel from pie chart
    plt.show()

def analyze_top_countries(df, top_n=5):
    """Analyzes and visualizes top countries of players."""
    print("\n--- Top Countries ---")
    country_counts = df['Country'].value_counts()
    print(f"Top {top_n} countries by player count:")
    print(country_counts.head(top_n))

    plt.figure(figsize=(12, 7))
    country_counts.head(top_n).plot(kind='bar', color=sns.color_palette("viridis", top_n))
    plt.title(f'Top {top_n} Countries by Player Count')
    plt.xlabel('Country')
    plt.ylabel('Number of Players')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def analyze_most_played_games(df):
    """Analyzes and visualizes most played games by total hours."""
    print("\n--- Most Played Games ---")
    # By total hours played
    game_hours = df.groupby('GamePlayed')['HoursPlayed'].sum().sort_values(ascending=False)
    print("Games by Total Hours Played:")
    print(game_hours)

    plt.figure(figsize=(14, 8))
    game_hours.head(10).plot(kind='barh', color=sns.color_palette("magma", 10))
    plt.title('Top Games by Total Hours Played')
    plt.xlabel('Total Hours Played')
    plt.ylabel('Game Title')
    plt.gca().invert_yaxis() # To show the top game at the top
    plt.tight_layout()
    plt.grid(axis='x', alpha=0.75)
    plt.show()
    
    return game_hours.index[0] # Return the most played game title

def analyze_correlation_for_top_game(df, top_game):
    """
    Tries to find correlations or insights for why the top game is popular.
    This is speculative and based on the available mock data.
    """
    print(f"\n--- Analysis for Top Game: {top_game} ---")

    top_game_df = df[df['GamePlayed'] == top_game]
    
    # 1. Demographics of players of the top game
    print(f"\nDemographics for players of {top_game}:")
    avg_age_top_game = top_game_df['Age'].mean()
    print(f"  Average Age: {avg_age_top_game:.2f} years")
    
    gender_dist_top_game = top_game_df['Gender'].value_counts(normalize=True) * 100
    print("  Gender Distribution:")
    print(gender_dist_top_game)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(top_game_df['Age'], bins=15, kde=True, color='skyblue')
    plt.title(f'Age Distribution for {top_game} Players')
    plt.xlabel('Age')
    plt.ylabel('Number of Players')
    
    plt.subplot(1, 2, 2)
    gender_dist_top_game.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title(f'Gender Distribution for {top_game} Players')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    # 2. Top countries for the top game
    top_game_country_counts = top_game_df['Country'].value_counts().head(5)
    print(f"\nTop 5 countries for {top_game}:")
    print(top_game_country_counts)
    
    plt.figure(figsize=(10, 6))
    top_game_country_counts.plot(kind='bar', color=sns.color_palette("coolwarm", 5))
    plt.title(f'Top 5 Countries for {top_game} Players')
    plt.xlabel('Country')
    plt.ylabel('Number of Players')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 3. Genre correlation (simplified)
    top_game_genre = top_game_df['PreferredGenre'].iloc[0] # All entries will have same genre for this game
    print(f"\n{top_game} is a(n) {top_game_genre} game.")

    # Speculative reasons based on mock data patterns:
    print("\nPotential (Speculative) Reasons for Popularity:")
    if top_game_genre in ['Action RPG', 'Tactical FPS', 'Tactical Shooter']:
        print(f"- The engaging nature of {top_game_genre}s often leads to higher playtime per player.")
    if avg_age_top_game < df['Age'].mean() - 2: # Significantly younger
        print(f"- Its appeal to a younger demographic (avg age {avg_age_top_game:.2f}) might contribute to its high engagement.")
    elif avg_age_top_game > df['Age'].mean() + 2: # Significantly older
        print(f"- It might resonate more with a mature audience (avg age {avg_age_top_game:.2f}), who may have more dedicated playtime.")
    else:
        print(f"- It has a broad appeal across various age groups (avg age {avg_age_top_game:.2f} is close to overall average).")

    # Compare gender distribution to overall
    overall_male_percentage = (df['Gender'].value_counts(normalize=True) * 100).get('Male', 0)
    top_game_male_percentage = gender_dist_top_game.get('Male', 0)
    if abs(top_game_male_percentage - overall_male_percentage) > 5: # More than 5% difference
        if top_game_male_percentage > overall_male_percentage:
            print(f"- The game seems to be particularly popular among Male players ({top_game_male_percentage:.1f}%) compared to the overall player base.")
        else:
             print(f"- The game shows a more diverse gender appeal or is particularly popular among Female/Non-Binary players compared to the overall player base.")
    
    print("\nDisclaimer: These are correlations based on mock data. Real-world analysis would require much richer datasets including gameplay telemetry, player surveys, social sentiment, marketing data, etc.")


# --- Main Execution ---
if __name__ == "__main__":
    # Generate or load data
    # To use a real CSV:
    # try:
    #     player_df = pd.read_csv('your_ubisoft_dataset.csv')
    # except FileNotFoundError:
    #     print("Dataset file not found. Generating mock data instead.")
    #     player_df = generate_mock_data(num_records=20000) # Generate more for better visuals
    # except Exception as e:
    #     print(f"Error loading dataset: {e}. Generating mock data instead.")
    #     player_df = generate_mock_data(num_records=20000)
        
    print("Generating mock data for demonstration...")
    player_df = generate_mock_data(num_records=50000) # Using 50k records for better distributions
    print(f"\nGenerated {len(player_df)} mock player records.")
    print("First 5 records:")
    print(player_df.head())

    # Set a nicer style for plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'w' # white background for saved figures

    # Perform analysis
    analyze_demographics(player_df)
    analyze_top_countries(player_df, top_n=5)
    most_played_game = analyze_most_played_games(player_df)
    
    if most_played_game:
        analyze_correlation_for_top_game(player_df, most_played_game)

    print("\nAnalysis Complete.")